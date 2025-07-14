import os
import torch
import argparse
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from models.CamoEval_enhanced import EnhancedDegreeNet as Network
from models.dataset_sem import get_loader, test_dataset, get_test_loader


def compute_mae(pred, gt):
    return np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)))


def compute_level_accuracy(pred, gt, levels=5):
    """Compute accuracy based on discretized levels"""
    # Convert scores to levels (0-4 for 5 levels)
    pred_levels = np.floor(pred * levels / 10).clip(0, levels - 1)
    gt_levels = np.floor(gt * levels / 10).clip(0, levels - 1)
    return np.mean(pred_levels == gt_levels)


def compute_threshold_accuracy(pred, gt, thresholds=[0.5, 1.0]):
    """Compute threshold accuracy for multiple thresholds"""
    errors = np.abs(pred - gt)
    acc_dict = {}
    for t in thresholds:
        acc_dict[t] = np.mean(errors <= t)
    return acc_dict


def save_prediction_gray(pred, mask, name, save_dir, levels=10):
    """
    Save prediction map as grayscale image with discrete levels
    Args:
        pred: prediction tensor [1, 1, H, W]
        mask: mask tensor [1, 1, H, W]
        name: image name
        save_dir: directory to save image
        levels: number of discrete levels (0-10)
    """
    # Convert to numpy arrays
    pred_np = pred.squeeze().cpu().numpy()  # [H, W]
    mask_np = mask.squeeze().cpu().numpy()  # [H, W]

    # Clip and round to nearest integer for discrete levels
    pred_discrete = np.round(pred_np.clip(0, levels - 1))

    # Create grayscale mapping (0-255)
    gray_values = np.linspace(0, 255, levels, dtype=np.uint8)
    gray_map = np.zeros_like(pred_discrete, dtype=np.uint8)

    # Apply grayscale mapping only to masked regions
    for level in range(levels):
        gray_map[(pred_discrete == level) & (mask_np > 0)] = gray_values[level]

    # Save image
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(gray_map).save(os.path.join(save_dir, f"{name}.png"))


def save_fixation_gray(fix_pred, mask, name, save_dir):
    """Save fixation map with enhanced semantic visualization"""

    fix_pred = fix_pred.squeeze().cpu().numpy()  # [H, W]
    mask = mask.squeeze().cpu().numpy()  # [H, W]

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


def test(test_loader, model, save_path):
    """
    Test function for evaluating the model and saving predictions
    to txt files corresponding to image names.
    """
    model.eval()

    # Metrics initialization
    total_mae = 0.0
    total_level_acc = 0.0
    total_acc_t = {0.5: 0.0, 1.0: 0.0}  # For threshold accuracy
    count = 0
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for i, (images, mask, gt, seg_label, names) in enumerate(test_loader, start=1):
            images = images.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True).float()
            seg_label = seg_label.cuda(non_blocking=True)

            outputs = model.forward_enhanced(images, mask, training_mode=True)
            res = outputs['target_pred']

            for b in range(res.shape[0]):
                res_pred = float(res[b].cpu().numpy())  # Convert to scalar
                res_gt = float(gt[b].cpu().numpy())  # Convert to scalar
                img_name = os.path.splitext(names[b])[0]

                # Store for global metrics
                all_preds.append(res_pred)
                all_gts.append(res_gt)

                # Compute per-sample metrics
                total_mae += compute_mae(np.array([res_pred]), np.array([res_gt]))
                total_level_acc += compute_level_accuracy(np.array([res_pred]), np.array([res_gt]))
                acc_t = compute_threshold_accuracy(np.array([res_pred]), np.array([res_gt]))
                for t in acc_t:
                    total_acc_t[t] += acc_t[t]
                count += 1

            # Save predictions and visualization
            for idx, name in enumerate(names):
                # Save score as text file
                txt_file_path = os.path.join(save_path, "scores", f"{name}.txt")
                os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
                np.savetxt(txt_file_path,
                           np.array([outputs['target_pred'][idx].cpu().numpy()]),
                           fmt='%.3f')

                # Save visualization
                if 'pred' in outputs:  # Check if model outputs pixel-wise predictions
                    vis_dir = os.path.join(save_path, "Visualization_sem")
                    save_prediction_gray(outputs['pred'][idx].unsqueeze(0),
                                         mask[idx].unsqueeze(0),
                                         name,
                                         vis_dir)

                if 'fixation' in outputs:  # Check if model outputs pixel-wise fixation
                    vis_dir = os.path.join(save_path, "Visualization_fixation")
                    save_fixation_gray(outputs['fixation'][idx].unsqueeze(0),
                                         mask[idx].unsqueeze(0),
                                         name,
                                         vis_dir)

    # Compute final metrics
    MAE = total_mae / count if count > 0 else 0.0
    Level_Acc = total_level_acc / count if count > 0 else 0.0
    Acc_t = {t: total_acc_t[t] / count for t in total_acc_t}

    # Compute global metrics
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    R2 = r2_score(all_gts, all_preds)
    Spearman = spearmanr(all_gts, all_preds)[0]

    return {
        'MAE': MAE,
        'R2': R2,
        'Spearman': Spearman,
        'Level_Acc': Level_Acc,
        'Acc_t': Acc_t
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='testing batch size')
    parser.add_argument('--testsize', type=int, default=352, help='testing dataset size')
    parser.add_argument('--model_path', type=str, default='./log/MyTrain_Enhanced/Net_epoch_best.pth',
                        help='path to the trained model')
    parser.add_argument('--test_root', type=str, default='./dataset/TestDataset/COD10K',
                        help='root directory for test dataset')
    parser.add_argument('--save_path', type=str, default='./exp_result/Test_Enhanced',
                        help='path to save prediction txt files')
    parser.add_argument('--gpu_id', type=str, default='1', help='which GPU to use for testing')
    opt = parser.parse_args()

    # Set the device for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True

    # Load the model
    model = Network(channel=128)
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    model.cuda()

    # Load the test data
    test_loader = get_test_loader(image_root=os.path.join(opt.test_root, 'Imgs_full/'),
                                  mask_root=os.path.join(opt.test_root, 'Mask/'),
                                  gt_root=os.path.join(opt.test_root, 'Ranking_proj/'),
                                  batchsize=opt.batchsize,
                                  testsize=opt.testsize,
                                  num_workers=4)

    # Prepare save path
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    os.makedirs(os.path.join(opt.save_path, "scores"), exist_ok=True)

    # Test the model
    metrics = test(test_loader, model, opt.save_path)

    # Print results
    print("\n===== Evaluation Results =====")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"Spearman: {metrics['Spearman']:.4f}")
    print(f"Level Accuracy (5 levels): {metrics['Level_Acc'] * 100:.2f}%")
    for t in sorted(metrics['Acc_t'].keys()):
        print(f"Acc_{t}: {metrics['Acc_t'][t] * 100:.2f}%")