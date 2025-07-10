import os
import torch
import argparse
import numpy as np
import imageio
from tqdm import tqdm
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.dataset import get_test_loader
from models.Camo_Eval import UEDGNet as Network
import logging


def test(test_loader, model, save_path):
    """
    Test function for evaluating the model and saving predictions
    to txt files corresponding to image names.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, mask, gt, names) in enumerate(test_loader):  # Now includes names
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)

            # Get model predictions
            res = model(image, mask)  # Model output

            pred_labels = res.argmax(dim=1)  # Get predicted class labels

            # Calculate accuracy
            correct += (pred_labels == gt-1).sum().item()
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
    print(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='testing batch size')
    parser.add_argument('--testsize', type=int, default=352, help='testing dataset size')
    parser.add_argument('--model_path', type=str, default='./log/MyTrain/Net_epoch_best.pth',
                        help='path to the trained model')
    parser.add_argument('--test_root', type=str, default='./dataset/TestDataset/COD10K',
                        help='root directory for test dataset')
    parser.add_argument('--save_path', type=str, default='./exp_result/Test', help='path to save prediction txt files')
    parser.add_argument('--gpu_id', type=str, default='1', help='which GPU to use for testing')
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
