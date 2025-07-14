import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


def cv_random_flip(img, mask):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def randomCrop(image, mask):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), mask.crop(random_region)


def randomRotation(image, mask):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        mask = mask.rotate(random_angle, mode)
    return image, mask


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, mask_root, gt_root, trainsize):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.masks = [mask_root + f for f in os.listdir(mask_root) if f.endswith('.jpg')
                      or f.endswith('.png')]
        self.gts = [gt_root + f.split('.')[0] + '.txt' for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]

        # sorted files
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        self.gts = sorted(self.gts)

        # filter matching degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)
        print('>>> training/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        mask = self.binary_loader(self.masks[index])
        gt = self.load_gt(self.gts[index])

        # data augmentation
        image, mask = cv_random_flip(image, mask)
        image, mask = randomCrop(image, mask)
        image, mask = randomRotation(image, mask)

        image = colorEnhance(image)
        mask = randomPeper(mask)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        # prepare segmentation label
        seg_label = self.prepare_segmentation_label(mask.squeeze(0), gt)  # mask.squeeze(0) to (H, W)

        return image, mask, gt, seg_label

    def prepare_segmentation_label(self, mask, gt):
        semantic_label = torch.zeros_like(mask, dtype=torch.long)
        # Convert float score to integer class (0-10 range)
        gt_class = int(round(gt))
        semantic_label[mask == 1] = gt_class
        return semantic_label

    def load_gt(self, gt_path):
        with open(gt_path, 'r') as f:
            gt = f.read().strip()
        return float(gt)

    def filter_files(self):
        assert len(self.images) == len(self.masks) and len(self.masks) == len(self.images)
        images = []
        masks = []
        gts = []
        for img_path, mask_path, gt_path in zip(self.images, self.masks, self.gts):
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            if img.size == mask.size:
                images.append(img_path)
                masks.append(mask_path)
                gts.append(gt_path)
        self.images = images
        self.masks = masks
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, mask_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = CamObjDataset(image_root, mask_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset(data.Dataset):
    def __init__(self, image_root, mask_root, gt_root, testsize):
        self.testsize = testsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.masks = sorted([mask_root + f for f in os.listdir(mask_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts = sorted([gt_root + f.split('.')[0] + '.txt' for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),  # Only resize without other transforms
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        mask = self.binary_loader(self.masks[index])
        gt = self.load_gt(self.gts[index])

        # Get the file name from the image path (excluding the extension)
        name = os.path.splitext(os.path.basename(self.images[index]))[0]

        # Apply the transformations
        image = self.transform(image)
        mask = self.mask_transform(mask)

        # prepare segmentation label
        seg_label = self.prepare_segmentation_label(mask.squeeze(0), gt)  # mask.squeeze(0) to (H, W)

        return image, mask, gt, seg_label, name  # Return the image, mask, gt, and file name


    def prepare_segmentation_label(self, mask, gt):
        semantic_label = torch.zeros_like(mask, dtype=torch.long)
        # Convert float score to integer class (0-10 range)
        gt_class = int(round(gt))
        semantic_label[mask == 1] = gt_class
        return semantic_label

    def __len__(self):
        return len(self.images)

    def load_gt(self, gt_path):
        with open(gt_path, 'r') as f:
            gt = f.read().strip()
        return float(gt)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

def get_test_loader(image_root, mask_root, gt_root, batchsize, testsize, num_workers=4, pin_memory=True):
    dataset = test_dataset(image_root, mask_root, gt_root, testsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=False,  
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader