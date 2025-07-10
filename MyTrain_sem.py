# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
# torch libraries
import os
import logging
from sched import scheduler
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
# customized libraries
import eval.metrics as Measure
#from models.Dual_Encoder_CamoEval_Sem_newmask import UEDGNet as Network
#from models.Simple_CamoEval import UEDGNet as Network
#from models.UEDGNet_iterative_pvt_antiartifact_laplace import UEDGNet as Network
from models.Camo_Eval import UEDGNet as Network
from models.utils import clip_gradient
from models.dataset_sem import get_loader, test_dataset, get_test_loader

sem_gray_map = {
    0: 0,
    1: 51,
    2: 102,
    3: 153,
    4: 204,
    5: 255
}

def save_semantic_gray(sem_pred, name, save_dir):
    gray = np.zeros_like(sem_pred, dtype=np.uint8)
    for k, v in sem_gray_map.items():
        gray[sem_pred == k] = v

    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(gray).save(os.path.join(save_dir, f"{name}.png"))

def weighted_segmentation_loss(pred_logits, target_labels):

    B, C, H, W = pred_logits.shape
    device = pred_logits.device

    # Step 1: Cross entropy loss per pixel
    ce_loss = F.cross_entropy(pred_logits, target_labels, reduction='none')  # (B, H, W)

    pred_probs = F.softmax(pred_logits, dim=1)  # (B, 6, H, W)
    pred_classes = pred_probs.argmax(dim=1)     # (B, H, W)

    # Step 3: Similarity prior matrix (6x6)
    similarity_prior = torch.tensor([
        [0.1, 0.6, 0.6, 0.6, 0.6, 0.6],  # 0:BG
        [0.6, 0.1, 0.2, 0.3, 0.4, 0.5],  # 1:most camo
        [0.6, 0.2, 0.1, 0.2, 0.3, 0.4],  # 2
        [0.6, 0.3, 0.2, 0.1, 0.2, 0.3],  # 3
        [0.6, 0.4, 0.3, 0.2, 0.1, 0.2],  # 4
        [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],  # 5:most sal
    ], device=device)

    penalty_weights = similarity_prior[target_labels, pred_classes]  # (B, H, W)

    weighted_loss = ce_loss * penalty_weights

    final_loss = weighted_loss.mean()

    return final_loss

def classification_loss(pred, gt):
    """
    Cross entropy loss for classification (5 classes)
    """
    loss = F.cross_entropy(pred, gt)
    return loss


def train(train_loader, model, optimizer, epoch, save_path, writer):
    global step
    model.train()
    model.cuda()
    loss_all = 0
    epoch_step = 0

    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()

    try:
        for i, (images, mask, gts, seg_label) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            seg_label = seg_label.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                sem, preds = model(images, mask)
                loss_sem = weighted_segmentation_loss(sem,seg_label)
                loss_cls = classification_loss(preds, gts)
                loss = loss_sem + loss_cls

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #del images, gts, preds
            #torch.cuda.empty_cache()

            step += 1
            epoch_step += 1
            loss_all += loss.detach().cpu().item()

            if i % 100 == 0 or i == total_step or i == 1:
                print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], '
                      f'Step [{i:04d}/{total_step}], Loss: {loss.item():.4f}')
                logging.info(f'[Train] Epoch [{epoch:03d}/{opt.epoch:03d}], '
                             f'Step [{i:04d}/{total_step}], Loss: {loss.item():.4f}')

                #writer.add_scalar('Loss_total', loss.item(), global_step=step)

        loss_all /= epoch_step
        logging.info(f'[Train] Epoch [{epoch:03d}/{opt.epoch:03d}], Avg Loss: {loss_all:.4f}')
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if epoch % 1 == 0:
            #torch.save(model.state_dict(), os.path.join(save_path, f'Net_epoch_{epoch}.pth'))
            torch.save(model.state_dict(), os.path.join(save_path, f'Net_epoch_last.pth'))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: Saving model before exit...')
        #torch.save(model.state_dict(), os.path.join(save_path, f'Net_epoch_{epoch + 1}.pth'))
        torch.save(model.state_dict(), os.path.join(save_path, f'Net_epoch_last.pth'))
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    Validation function
    """
    global best_metric_dict, best_score, best_epoch

    if epoch == 1:
        best_metric_dict = {'ACC': 0.0}  # Initialize with default value

    ACC = Measure.Accuracy()

    metrics_dict = dict()

    model.eval()

    save_dir = 'log/result'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():

        for i, (image, mask, gt, seg_label, name) in enumerate(test_loader):

            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            seg_label = seg_label.cuda(non_blocking=True)

            sem, res = model(image, mask)
            sem = F.softmax(sem, dim=1)  # (B, 6, H, W)
            sem = sem.argmax(dim=1)

            for b in range(sem.shape[0]):
                sem_pred = sem[b].cpu().numpy()  # (H, W)
                img_name = os.path.splitext(name[b])[0]
                save_semantic_gray(sem_pred, img_name, save_dir)


            res = F.softmax(res, dim=1)
            pred_labels = res.argmax(dim=1)

            try:
                ACC.step(pred=pred_labels, gt=gt)
            except Exception as e:
                print(f"Error at sample {i}: {e}")

        metrics_dict.update(ACC=ACC.get_results()['accuracy'])

        cur_score = metrics_dict['ACC']
        writer.add_scalar('ACC', cur_score, global_step=epoch)

        if epoch == 1:
            best_score = cur_score
            print(f'[Cur Epoch: {epoch}] Metrics (ACC={metrics_dict["ACC"]})')
            logging.info(f'[Cur Epoch: {epoch}] Metrics (ACC={metrics_dict["ACC"]})')
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print(f'>>> Saved best model at epoch {epoch}')
            else:
                print('>>> No improvement, continue training...')
            print(f'[Current Epoch {epoch}] ACC={metrics_dict["ACC"]}\n'
                  f'[Best Epoch {best_epoch}] ACC={best_metric_dict["ACC"]}')
            logging.info(f'[Current Epoch {epoch}] ACC={metrics_dict["ACC"]}\n'
                         f'[Best Epoch {best_epoch}] ACC={best_metric_dict["ACC"]}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')  # 16
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--model', type=str, default='UEDGNet', help='main model')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='./dataset/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./dataset/TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--save_path', type=str, default='./log/MyTrain/',
                        help='the path to save model and log')
    opt = parser.parse_args()

    # set the device for training
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

    model = Network(channel=64)
    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs_full/',
                              mask_root=opt.train_root + 'Mask/',
                              gt_root=opt.train_root + 'Ranking/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)

    val_loader = get_test_loader(image_root=opt.val_root + 'Imgs_full/',
                                 mask_root=opt.val_root + 'Mask/',
                                 gt_root=opt.val_root + 'Ranking/',
                                 batchsize=1,
                                 testsize=opt.trainsize,
                                 num_workers=4)

    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_score = 0
    best_epoch = 0

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print(">>> start train...")
    for epoch in range(1, opt.epoch):
        # schedule
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        # train
        train(train_loader, model, optimizer, epoch, save_path, writer)
        #if epoch > opt.epoch//2+50:
        if epoch > 0:
            # validation
            val(val_loader, model, epoch, save_path, writer)
