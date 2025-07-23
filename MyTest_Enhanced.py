import os
import torch
import argparse
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from thop import profile  # For FLOPs and parameters calculation
from torch.utils.data import DataLoader

from models.CamoEval_enhanced_final1 import EnhancedDegreeNet as Network
from models.dataset_sem import get_loader, test_dataset, get_test_loader


def compute_mae(pred, gt):
    return np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)))


def compute_level_accuracy(pred, gt, levels=5):
    pred_levels = np.floor((pred - 1) / 2).clip(0, levels - 1)
    gt_levels = np.floor((gt - 1) / 2).clip(0, levels - 1)
    return np.mean(pred_levels == gt_levels)


def compute_threshold_accuracy(pred, gt, thresholds=[0.5, 1.0, 2.0]):
    errors = np.abs(pred - gt)
    acc_dict = {}
    for t in thresholds:
        acc_dict[t] = np.mean(errors <= t)
    return acc_dict


def save_prediction_gray(pred, mask, name, save_dir, levels=10):
    pred_np = pred.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_discrete = np.round(pred_np.clip(0, levels - 1))

    gray_values = np.linspace(0, 255, levels, dtype=np.uint8)
    gray_map = np.zeros_like(pred_discrete, dtype=np.uint8)

    for level in range(levels):
        gray_map[(pred_discrete == level) & (mask_np > 0)] = gray_values[level]

    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(gray_map).save(os.path.join(save_dir, f"{name}.png"))


def save_fixation_gray(fix_pred, mask, name, save_dir):
    fix_pred = fix_pred.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0).astype(np.float32)
    masked_fixation = fix_pred * mask

    if masked_fixation[mask > 0].size > 0:
        min_val = masked_fixation[mask > 0].min()
        max_val = masked_fixation[mask > 0].max()
        if max_val > min_val:
            masked_fixation[mask > 0] = (masked_fixation[mask > 0] - min_val) / (max_val - min_val)

    vis_img = (masked_fixation * 255).astype(np.uint8)
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(vis_img).save(os.path.join(save_dir, f"{name}.png"))


def save_part_attention(part_attention, mask, name, save_dir):
    """
    Save part attention as a single color image where each pixel is colored
    according to its argmax part class (only within mask region)
    Args:
        part_attention: [1, num_parts, H, W] tensor or numpy array (softmax probabilities)
        mask: [1, 1, H, W] tensor
        name: image name
        save_dir: directory to save images
    """
    # Ensure part_attention is a tensor
    if isinstance(part_attention, np.ndarray):
        part_attention = torch.from_numpy(part_attention)

    # Ensure mask is a tensor
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    # First upsample while still in tensor form
    if part_attention.shape[2:] != mask.shape[2:]:
        part_attention = torch.nn.functional.interpolate(
            part_attention.float(),
            size=mask.shape[2:],
            mode='bilinear',
            align_corners=False
        )

    # Convert to numpy arrays
    part_attention = part_attention.squeeze(0).cpu().numpy()  # [num_parts, H, W]
    mask = mask.squeeze().cpu().numpy()  # [H, W]

    # Get argmax class for each pixel
    part_classes = np.argmax(part_attention, axis=0)  # [H, W]

    # Define a colormap (6 distinct colors for 6 parts)
    colors = np.array([
        [255, 0, 0],  # Red - Part 0
        [0, 255, 0],  # Green - Part 1
        [0, 0, 255],  # Blue - Part 2
        [255, 255, 0],  # Yellow - Part 3
        [255, 0, 255],  # Magenta - Part 4
        [0, 255, 255],  # Cyan - Part 5
    ], dtype=np.uint8)

    # Create RGB image
    h, w = part_classes.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Apply colors only to masked regions
    for part_idx in range(6):
        color_img[(part_classes == part_idx) & (mask > 0)] = colors[part_idx]

    # Save image
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(color_img).save(os.path.join(save_dir, f"{name}.png"))


def calculate_flops_params(model, input_size=(1, 3, 352, 352)):

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = 0
    try:

        dummy_image = torch.randn(*input_size).cuda()
        dummy_mask = torch.randn(input_size[0], 1, input_size[2], input_size[3]).cuda()

        flops_counter = FlopCountAnalysis(model, (dummy_image, dummy_mask))
        flops = flops_counter.total()
    except ImportError:
        print("Warning: fvcore not installed, FLOPs will be 0. Install with: pip install fvcore")
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")

    return flops, params

def test(test_loader, model, save_path):
    model.eval()

    total_mae = 0.0
    total_level_acc = 0.0
    total_acc_t = {0.5: 0.0, 1.0: 0.0, 2.0: 0.0}
    count = 0
    all_preds = []
    all_gts = []

    # For inference time measurement
    total_time = 0.0
    warmup = 5  # Warmup iterations to avoid initial overhead

    # Calculate FLOPs and parameters first
    try:
        flops, params = calculate_flops_params(model, input_size=(1, 3, opt.testsize, opt.testsize))
    except Exception as e:
        print(f"Could not calculate FLOPs and parameters: {e}")
        flops, params = 0, 0

    with torch.no_grad():
        for i, (images, mask, gt, seg_label, names) in enumerate(test_loader, start=1):
            images = images.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True).float()
            seg_label = seg_label.cuda(non_blocking=True)

            # Measure inference time
            if i > warmup:  # Skip warmup iterations
                torch.cuda.synchronize()  # Wait for all kernels to finish
                start_time = time.time()

            outputs = model.forward_enhanced(images, mask, training_mode=True)

            if i > warmup:
                torch.cuda.synchronize()  # Wait for all kernels to finish
                end_time = time.time()
                total_time += (end_time - start_time)

            res = outputs['target_pred']

            for b in range(res.shape[0]):
                res_pred = float(res[b].cpu().numpy())
                res_gt = float(gt[b].cpu().numpy())
                img_name = os.path.splitext(names[b])[0]

                all_preds.append(res_pred)
                all_gts.append(res_gt)

                total_mae += compute_mae(np.array([res_pred]), np.array([res_gt]))
                # total_level_acc += compute_level_accuracy(np.array([res_pred]), np.array([res_gt]))
                acc_t = compute_threshold_accuracy(np.array([res_pred]), np.array([res_gt]))
                for t in acc_t:
                    total_acc_t[t] += acc_t[t]
                count += 1

            # Save predictions and visualization
            for idx, name in enumerate(names):
                # Save score
                txt_file_path = os.path.join(save_path, "scores", f"{name}.txt")
                os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
                np.savetxt(txt_file_path,
                           np.array([outputs['target_pred'][idx].cpu().numpy()]),
                           fmt='%.3f')

                # Save visualizations
                if 'pred' in outputs:
                    vis_dir = os.path.join(save_path, "Visualization_sem")
                    save_prediction_gray(outputs['pred'][idx].unsqueeze(0),
                                         mask[idx].unsqueeze(0),
                                         name,
                                         vis_dir)

                if 'fixation' in outputs:
                    vis_dir = os.path.join(save_path, "Visualization_fixation")
                    save_fixation_gray(outputs['fixation'][idx].unsqueeze(0),
                                       mask[idx].unsqueeze(0),
                                       name,
                                       vis_dir)

                # part_attention
                if 'part_attention' in outputs:
                    vis_dir = os.path.join(save_path, "part_attention")
                    save_part_attention(outputs['part_attention'][idx].unsqueeze(0),
                                        mask[idx].unsqueeze(0),
                                        name,
                                        vis_dir)

    # Compute metrics
    MAE = total_mae / count if count > 0 else 0.0
    # Level_Acc = total_level_acc / count if count > 0 else 0.0
    Acc_t = {t: total_acc_t[t] / count for t in total_acc_t}

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    R2 = r2_score(all_gts, all_preds)
    Spearman = spearmanr(all_gts, all_preds)[0]

    # Calculate average inference time (skip warmup iterations)
    avg_time = total_time / (len(test_loader) - warmup) if (len(test_loader) - warmup) > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0



    return {
        'MAE': MAE,
        'R2': R2,
        'Spearman': Spearman,
        'Acc_t': Acc_t,
        'Inference_time': avg_time,
        'FPS': fps,
        'FLOPs': flops,
        'Params': params
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='testing batch size')
    parser.add_argument('--testsize', type=int, default=352, help='testing dataset size')
    parser.add_argument('--model_path', type=str, default='./log/MyTrain_Enhanced_final1/Net_epoch_best.pth',
                        help='path to the trained model')
    parser.add_argument('--test_root', type=str, default='./dataset/TestDataset/COD10K',
                        help='root directory for test dataset')
    parser.add_argument('--save_path', type=str, default='./exp_result/Test_Enhanced_final1_973',
                        help='path to save prediction txt files')
    parser.add_argument('--gpu_id', type=str, default='1', help='which GPU to use for testing')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True

    model = Network(channel=128)
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    model.cuda()

    test_loader = get_test_loader(image_root=os.path.join(opt.test_root, 'Imgs_full/'),
                                  mask_root=os.path.join(opt.test_root, 'Mask/'),
                                  gt_root=os.path.join(opt.test_root, 'Ranking_proj/'),
                                  batchsize=opt.batchsize,
                                  testsize=opt.testsize,
                                  num_workers=4)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    os.makedirs(os.path.join(opt.save_path, "scores"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_path, "part_attention"), exist_ok=True)

    metrics = test(test_loader, model, opt.save_path)

    print("\n===== Evaluation Results =====")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"Spearman: {metrics['Spearman']:.4f}")
    # print(f"Level Accuracy (5 levels): {metrics['Level_Acc'] * 100:.2f}%")
    for t in sorted(metrics['Acc_t'].keys()):
        print(f"Acc_{t}: {metrics['Acc_t'][t] * 100:.2f}%")

    print("\n===== Model Efficiency Metrics =====")
    print(f"Average Inference Time: {metrics['Inference_time'] * 1000:.2f} ms")
    print(f"FPS: {metrics['FPS']:.2f}")
    print(f"FLOPs: {metrics['FLOPs'] / 1e9:.2f} G")
    print(f"Parameters: {metrics['Params'] / 1e6:.2f} M")