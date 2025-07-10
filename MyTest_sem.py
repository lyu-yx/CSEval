import os
import torch
import argparse
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.dataset_sem import get_test_loader
#from models.Dual_Encoder_CamoEval_Sem_newmask import UEDGNet as Network
from models.Dual_Encoder_CamoEval_Sem import UEDGNet as Network
#from models.Camo_Eval import UEDGNet as Network
#from models.Simple_CamoEval import UEDGNet as Network
import logging

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

def compute_mae(pred, gt):
    return np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)))

def test(test_loader, model, save_path):
    """
    Test function for evaluating the model and saving predictions
    to txt files corresponding to image names.
    """
    model.eval()
    correct = 0
    total = 0

    total_mae = 0.0
    count = 0
    with torch.no_grad():
        for i, (image, mask, gt, seg_label, names) in enumerate(test_loader):
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            seg_label = seg_label.cuda(non_blocking=True)

            sem, res = model(image, mask)
            sem = F.softmax(sem, dim=1)  # (B, 6, H, W)
            sem = sem.argmax(dim=1)

            for name in enumerate(names):
                for b in range(sem.shape[0]):
                    sem_pred = sem[b].cpu().numpy()  # (H, W)
                    sem_gt = seg_label[b].cpu().numpy()
                    img_name = name[1]
                    save_semantic_gray(sem_pred, img_name, './exp_result/Test/sem')
                    total_mae += compute_mae(sem_pred, sem_gt)
                    count += 1

            res = F.softmax(res, dim=1)
            pred_labels = res.argmax(dim=1)

            # Calculate accuracy
            correct += (pred_labels == gt).sum().item()
            total += gt.numel()


            # Save results to corresponding txt files
            for idx, name in enumerate(names):
                # Save predicted labels as text file with the same name as the image
                txt_file_path = os.path.join(save_path, f"{name}.txt")
                np.savetxt(txt_file_path, np.array([pred_labels[idx].cpu().numpy()]), fmt='%d')  # Convert to array

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} batches")

    # Compute overall accuracy

    accuracy = 100 * correct / total
    MAE = total_mae / count if count > 0 else 0.0
    print(f"Accuracy: {accuracy:.2f}%",f"rMAE: {MAE:.2f}")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"rMAE: {MAE:.2f}")

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='testing batch size')
    parser.add_argument('--testsize', type=int, default=352, help='testing dataset size')
    parser.add_argument('--model_path', type=str, default='./log/MyTrain/Net_epoch_best_DualEncoder_sem.pth',
                        help='path to the trained model')
    parser.add_argument('--test_root', type=str, default='./dataset/TestDataset/COD10K',
                        help='root directory for test dataset')
    parser.add_argument('--save_path', type=str, default='./exp_result/Test', help='path to save prediction txt files')
    parser.add_argument('--gpu_id', type=str, default='0', help='which GPU to use for testing')
    opt = parser.parse_args()

    # Set the device for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True

    # Load the model
    model = Network(channel=64)  # Assuming the model is `Network`
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    model.cuda()

    # Load the test data
    test_loader = get_test_loader(image_root=os.path.join(opt.test_root, 'Imgs_full/'),
                                  mask_root=os.path.join(opt.test_root, 'Mask/'),
                                  gt_root=os.path.join(opt.test_root, 'Ranking/'),
                                  batchsize=opt.batchsize,
                                  testsize=opt.testsize,
                                  num_workers=4)


    # Prepare save path
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    # Test the model
    accuracy = test(test_loader, model, opt.save_path)
    print(f"Final Test Accuracy: {accuracy:.2f}%")
