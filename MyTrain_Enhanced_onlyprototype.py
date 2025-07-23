# Enhanced Degree Model training script with semantic discovery and domain adaptation (Default)
import os
import logging
import numpy as np
import sys

sys.path.append('')
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torchvision.utils import make_grid
from PIL import Image

# Evaluation metrics
import eval.metrics as Measure

# Enhanced models and components
from models.CamoEval_enhanced_onlyprototype import EnhancedDegreeNet as Network
from models.enhanced_components_onlyprototype import StreamlinedLoss
from models.utils import clip_gradient
from models.dataset_sem import get_loader, test_dataset, get_test_loader

# Semantic mapping for visualization (keeping original)
sem_gray_map = {
    0: 0, 1: 25, 2: 51, 3: 75, 4: 102, 5: 125,
    6: 153, 7: 175, 8: 204, 9: 225, 10: 255
}


def compute_mae(pred, gt):
    return np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)))


def save_semantic_gray(sem_pred, name, save_dir):
    gray = np.zeros_like(sem_pred, dtype=np.uint8)
    for k, v in sem_gray_map.items():
        gray[sem_pred == k] = v
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(gray).save(os.path.join(save_dir, f"{name}.png"))


def save_fixation_gray(fix_pred, mask, name, save_dir):
    """Save fixation map with enhanced semantic visualization"""
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
    """Save semantic part attention maps"""
    os.makedirs(save_dir, exist_ok=True)
    mask = (mask > 0).astype(np.float32)

    # Save each semantic part as a separate image
    for part_idx in range(part_attention.shape[0]):
        part_map = part_attention[part_idx] * mask
        if part_map.max() > part_map.min():
            part_map = (part_map - part_map.min()) / (part_map.max() - part_map.min())
        part_img = (part_map * 255).astype(np.uint8)
        Image.fromarray(part_img).save(os.path.join(save_dir, f"{name}_part_{part_idx}.png"))


def get_domain_label(gt_value):
    """Determine domain label based on ground truth value"""
    # COD typically ranges 0-5, SOD typically ranges 5-10
    return 0 if gt_value <= 5 else 1  # 0 for COD, 1 for SOD


def train(train_loader, model, optimizer, epoch, save_path, writer, criterion):
    global step
    model.train()
    model.cuda()
    loss_all = 0
    epoch_step = 0

    # Enhanced loss tracking
    loss_components = {
        'main_loss': 0, 'perceptual_loss': 0, 'semantic_loss': 0
    }

    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()

    try:
        for i, (images, mask, gts, seg_label) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True).float()

            seg_label = seg_label.cuda(non_blocking=True)

            # Generate domain labels based on ground truth values
            domain_labels = torch.tensor([get_domain_label(gt.item()) for gt in gts]).cuda()

            with torch.cuda.amp.autocast():
                # Enhanced model is default - use enhanced forward pass
                if hasattr(model, 'forward_enhanced'):
                    outputs = model.forward_enhanced(images, mask, training_mode=True)

                    # Enhanced loss computation
                    loss, loss_dict = criterion(outputs, gts, images, mask, domain_labels)

                    # Track individual loss components
                    for key in loss_components:
                        loss_components[key] += loss_dict.get(key, 0)

                else:
                    # Fallback for models without enhanced features
                    outputs = model(images, mask)

                    outputs = {'pred': outputs['pred'], 'target_pred': outputs['target_pred'],
                               'fixation': outputs['fixation'],
                               'pvt_features': None, 'part_attention': None, 'domain_logits': None}
                    loss, loss_dict = criterion(outputs, gts, images, mask, domain_labels)

            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            step += 1
            epoch_step += 1
            loss_all += loss.detach().cpu().item()

            if i % 100 == 0 or i == total_step or i == 1:
                print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], '
                      f'Step [{i:04d}/{total_step}], Loss: {loss.item():.4f}')
                # Enhanced features are default - always show detailed loss breakdown
                print(
                    f'  Main: {loss_dict["main_loss"]:.4f}, Perceptual: {loss_dict["perceptual_loss"]:.4f}, Semantic: {loss_dict["semantic_loss"]:.4f}')

                logging.info(f'[Train] Epoch [{epoch:03d}/{opt.epoch:03d}], '
                             f'Step [{i:04d}/{total_step}], Loss: {loss.item():.4f}')

        loss_all /= epoch_step
        logging.info(f'[Train] Epoch [{epoch:03d}/{opt.epoch:03d}], Avg Loss: {loss_all:.4f}')

        # Log enhanced metrics (always available with Enhanced Degree Model)
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        for key, value in loss_components.items():
            avg_value = value / epoch_step
            writer.add_scalar(f'Loss-{key}', avg_value, global_step=epoch)

        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'Net_epoch_last.pth'))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: Saving model before exit...')
        torch.save(model.state_dict(), os.path.join(save_path, f'Net_epoch_last.pth'))
        raise


def val(test_loader, model, epoch, save_path, writer):
    """Enhanced validation with semantic part visualization"""
    global best_metric_dict, best_score, best_epoch

    if epoch == 1:
        best_metric_dict = {'MAE': 0.0}

    # ACC = Measure.Accuracy()
    metrics_dict = dict()
    model.eval()

    # Enhanced save directories
    save_dir_fix = 'log/result_fix'
    save_dir_parts = 'log/result_parts'
    os.makedirs(save_dir_fix, exist_ok=True)
    os.makedirs(save_dir_parts, exist_ok=True)

    total_mae = 0.0
    count = 0

    with torch.no_grad():
        for i, (image, mask, gt, seg_label, names) in enumerate(test_loader):
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            # seg_label = seg_label.cuda(non_blocking=True)

            # Enhanced forward pass if available
            if hasattr(model, 'forward_enhanced'):
                outputs = model.forward_enhanced(image, mask, training_mode=False)
                if isinstance(outputs, dict):
                    sem = outputs['pred']
                    res = outputs['target_pred']
                    fix = outputs['fixation']
                    part_attention = outputs.get('part_attention', None)
                else:
                    sem, res, fix = outputs
                    part_attention = None
            else:
                sem, res, fix = model(image, mask)
                part_attention = None

            # Process semantic predictions

            fix = fix.squeeze(1)

            # Save results and part attention if available
            for b in range(res.shape[0]):
                res_pred = res[b].cpu().numpy()
                res_gt = gt[b].cpu().numpy()
                fix_pred = fix[b].cpu().numpy()
                mask_np = mask[b].squeeze().cpu().numpy()
                img_name = os.path.splitext(names[b])[0]

                total_mae += compute_mae(res_pred, res_gt)
                count += 1

                save_fixation_gray(fix_pred, mask_np, img_name, save_dir_fix)
                '''
                # Save semantic part attention if available
                if part_attention is not None:
                    part_att_np = part_attention[b].cpu().numpy()
                    save_part_attention(part_att_np, mask_np, img_name, save_dir_parts)
                '''

        MAE = total_mae / count if count > 0 else 0.0
        metrics_dict.update(MAE=MAE)

        cur_score = metrics_dict['MAE']
        writer.add_scalar('MAE', MAE, global_step=epoch)

        # Enhanced logging
        if epoch == 1:
            best_score = cur_score
            print(f'[Cur Epoch: {epoch}] Metrics (MAE={MAE:.4f})')
            logging.info(f'[Cur Epoch: {epoch}] Metrics (MAE={metrics_dict["MAE"]:.4f})')
        else:
            if cur_score < best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print(f'>>> Saved best model at epoch {epoch}')
            else:
                print('>>> No improvement, continue training...')

            print(f'[Current Epoch {epoch}] MAE={metrics_dict["MAE"]:.4f}')
            print(f'[Best Epoch {best_epoch}] MAE={best_metric_dict["MAE"]:.4f}')
            logging.info(f'[Current Epoch {epoch}] MAE={metrics_dict["MAE"]:.4f}')
            logging.info(f'[Best Epoch {best_epoch}] MAE={best_metric_dict["MAE"]:.4f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--model', type=str, default='EnhancedDegreeNet', help='main model')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='./dataset/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./dataset/TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--save_path', type=str, default='./log/MyTrain_Enhanced_onlyprototype/',
                        help='the path to save model and log')

    # Enhanced loss parameters (enhanced features are default)
    parser.add_argument('--alpha_perceptual', type=float, default=1, help='perceptual loss weight')
    parser.add_argument('--alpha_semantic', type=float, default=10, help='semantic loss weight')  # ori:0.1
    parser.add_argument('--disable_enhanced', action='store_true',
                        help='disable enhanced features (fallback to basic model)')

    opt = parser.parse_args()

    # Set device
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    cudnn.benchmark = True

    # Initialize model and loss (enhanced features by default)
    if opt.disable_enhanced:
        print("Using Basic Model (enhanced features disabled)")
        # Fallback: create a simple version - for now we'll still use enhanced but with minimal features
        model = Network(channel=64)
        criterion = StreamlinedLoss(
            alpha_perceptual=0.0,
            alpha_semantic=0.0
        )
    else:
        print("Using Enhanced Degree Model with Semantic Discovery and Domain Adaptation (Default)")
        model = Network(channel=128)  # Enhanced features by default
        criterion = StreamlinedLoss(
            alpha_perceptual=opt.alpha_perceptual,
            alpha_semantic=opt.alpha_semantic
        )

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    # Enhanced optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-5)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs_full/',
                              mask_root=opt.train_root + 'Mask/',
                              gt_root=opt.train_root + 'Ranking_proj/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)

    val_loader = get_test_loader(image_root=opt.val_root + 'Imgs_full/',
                                 mask_root=opt.val_root + 'Mask/',
                                 gt_root=opt.val_root + 'Ranking_proj/',
                                 batchsize=1,
                                 testsize=opt.trainsize,
                                 num_workers=4)

    total_step = len(train_loader)

    # Logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> Enhanced Degree Model training (default mode): network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_score = 0
    best_epoch = 0

    # Enhanced learning rate scheduler
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)

    print(">>> start training with Enhanced Degree Model...")
    for epoch in range(1, opt.epoch):
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))

        # Train with enhanced model (always available now)
        train(train_loader, model, optimizer, epoch, save_path, writer, criterion)

        if epoch > 0:
            val(val_loader, model, epoch, save_path, writer)