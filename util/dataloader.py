#!/usr/bin/env python
# coding: utf-8

import math
import os

import numpy as np
import torch

from torchvision import datasets, transforms


# generate train and validation image dataset
def get_image_datasets(args, scale_size, input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(scale_size),
            transforms.RandomRotation(args.random_rotate_degree),
            transforms.RandomResizedCrop(input_size,
                                         scale=args.random_resized_crop_scale,
                                         ratio=args.random_resized_crop_ratio),
            transforms.RandomHorizontalFlip(args.random_horizontal_flip),
            transforms.RandomVerticalFlip(args.random_vertical_flip),
            transforms.ColorJitter(
                brightness=args.jitter_brightness,
                contrast=args.jitter_contrast,
                saturation=args.jitter_saturation,
                hue=args.jitter_hue
            ),
            transforms.ToTensor(),
            transforms.Normalize(args.rgb_mean, args.rgb_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(args.rgb_mean, args.rgb_std)
        ])
    }

    # use Cutout
    if args.cutout:
        if args.mixup or args.ricap:
            pass
            # When using mixup or ricap, cutout is applied after batch creation for learning
        else:
            data_transforms['train'] = transforms.Compose([
                transforms.Resize(scale_size),
                transforms.RandomRotation(args.random_rotate_degree),
                transforms.RandomResizedCrop(input_size,
                                             scale=args.random_resized_crop_scale,
                                             ratio=args.random_resized_crop_ratio),
                transforms.RandomHorizontalFlip(args.random_horizontal_flip),
                transforms.RandomVerticalFlip(args.random_vertical_flip),
                transforms.ColorJitter(
                    brightness=args.jitter_brightness,
                    contrast=args.jitter_contrast,
                    saturation=args.jitter_saturation,
                    hue=args.jitter_hue
                ),
                transforms.ToTensor(),
                transforms.Normalize(args.rgb_mean, args.rgb_std),
                Cutout(n_holes=args.cutout_holes, length=args.cutout_length)
            ])
    else:
        args.cutout_holes = None
        args.cutout_length = None

    # use Random erasing
    if args.random_erasing:
        if args.mixup or args.ricap:
            pass
            # When using mixup or ricap, cutout is applied after batch creation for learning
        else:
            data_transforms['train'] = transforms.Compose([
                transforms.Resize(scale_size),
                transforms.RandomRotation(args.random_rotate_degree),
                transforms.RandomResizedCrop(input_size,
                                             scale=args.random_resized_crop_scale,
                                             ratio=args.random_resized_crop_ratio),
                transforms.RandomHorizontalFlip(args.random_horizontal_flip),
                transforms.RandomVerticalFlip(args.random_vertical_flip),
                transforms.ColorJitter(
                    brightness=args.jitter_brightness,
                    contrast=args.jitter_contrast,
                    saturation=args.jitter_saturation,
                    hue=args.jitter_hue
                ),
                transforms.ToTensor(),
                transforms.Normalize(args.rgb_mean, args.rgb_std),
                RandomErasing(p=args.random_erasing_p,
                              sl=args.random_erasing_sl,
                              sh=args.random_erasing_sh,
                              r1=args.random_erasing_r1,
                              r2=args.random_erasing_r2)
            ])
    else:
        args.random_erasing_p = None
        args.random_erasing_sl = None
        args.random_erasing_sh = None
        args.random_erasing_r1 = None
        args.random_erasing_r2 = None

    image_datasets = {
        x: ImageFolderWithPaths(os.path.join(args.data, x), data_transforms[x])
        for x in ['train', 'valid']
    }

    return image_datasets


def get_image_datasets_for_rgb_mean_and_std(args, scale_size, input_size):
    transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor()
        ])
    image_datasets = datasets.ImageFolder(args.data, transform=transform)
    return image_datasets


# generate train and validation dataloaders
def get_dataloader(args, scale_size, input_size):
    image_datasets = get_image_datasets(args, scale_size, input_size)

    train_num_classes = len(image_datasets['train'].classes)
    val_num_classes = len(image_datasets['valid'].classes)
    assert train_num_classes == val_num_classes, 'The number of classes for train and validation is different'

    train_class_names = image_datasets['train'].classes
    val_class_names = image_datasets['valid'].classes

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        image_datasets['valid'], batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, train_num_classes, train_class_names, \
        val_loader, val_num_classes, val_class_names


# https://arxiv.org/pdf/1708.04552.pdf
# modified from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask_value = img.mean()

        for n in range(self.n_holes):
            top = np.random.randint(0 - self.length // 2, h)
            left = np.random.randint(0 - self.length // 2, w)
            bottom = top + self.length
            right = left + self.length

            top = 0 if top < 0 else top
            left = 0 if left < 0 else left

            img[:, top:bottom, left:right].fill_(mask_value)

        return img


# https://arxiv.org/pdf/1708.04552.pdf
# modified from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class CutoutForBatchImages(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            batch images (Tensor): Tensor images of size (N, C, H, W).
        Returns:
            Tensor: Images with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask_value = img.mean()

        for i in range(img.size(0)):
            for n in range(self.n_holes):
                top = np.random.randint(0 - self.length // 2, h)
                left = np.random.randint(0 - self.length // 2, w)
                bottom = top + self.length
                right = left + self.length

                top = 0 if top < 0 else top
                left = 0 if left < 0 else left

                img[i, :, top:bottom, left:right].fill_(mask_value)

        return img


# https://arxiv.org/pdf/1708.04896.pdf
# modified from https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    p: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    r2: max aspect ratio
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with Random erasing.
        """
        if np.random.uniform(0, 1) > self.p:
            return img

        area = img.size()[1] * img.size()[2]
        for _attempt in range(100):
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, self.r2)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                return img

        return img


# https://arxiv.org/pdf/1708.04896.pdf
# modified from https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
class RandomErasingForBatchImages(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    p: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    r2: max aspect ratio
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):
        """
        Args:
            batch images (Tensor): Tensor images of size (N, C, H, W).
        Returns:
            Tensor: Images with Random erasing.
        """
        area = img.size()[2] * img.size()[3]
        for i in range(img.size(0)):

            if np.random.uniform(0, 1) > self.p:
                continue

            for _attempt in range(100):
                target_area = np.random.uniform(self.sl, self.sh) * area
                aspect_ratio = np.random.uniform(self.r1, self.r2)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[3] and h < img.size()[2]:
                    x1 = np.random.randint(0, img.size()[2] - h)
                    y1 = np.random.randint(0, img.size()[3] - w)
                    img[i, :, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, 3, h, w))
                    break

        return img


# taken from https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
