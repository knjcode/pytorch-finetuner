#!/usr/bin/env python
# coding: utf-8

import datetime
import numbers
import os
import shutil
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from cnn_finetune import make_model

from multiprocessing import cpu_count
from torchvision.transforms.functional import center_crop, hflip, vflip, resize


class IncrementalVariance(object):
    def __init__(self, avg=0, count=0, var=0):
        self.avg = avg
        self.count = count
        self.var = var

    def update(self, avg, count, var):
        delta = self.avg - avg
        m_a = self.var * (self.count - 1)
        m_b = var * (count - 1)
        M2 = m_a + m_b + delta ** 2 * self.count * count / (self.count + count)
        self.var = M2 / (self.count + count - 1)
        self.avg = (self.avg * self.count + avg * count) / (self.count + count)
        self.count = self.count + count

    @property
    def average(self):
        return self.avg

    @property
    def variance(self):
        return self.var

    @property
    def std(self):
        return np.sqrt(self.var)


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.item()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def print_batch(batch, epoch, current_num, total_num, ratio, speed, average_acc, average_loss, logger):
    logger.info('Epoch[{}] Batch[{}] [{}/{} ({:.0f}%)]\tspeed: {:.2f} samples/sec\taccuracy: {:.10f}\tloss: {:.10f}'.format(
        epoch, batch, current_num, total_num, ratio, speed, average_acc, average_loss))


def report(epoch, phase, loss_name, loss_avg, acc_name, acc_avg, logger, log_writer):
    logger.info("Epoch[{}] {}-accuracy: {}".format(epoch, phase, acc_avg))
    logger.info("Epoch[{}] {}-loss: {}".format(epoch, phase, loss_avg))
    if log_writer:
        log_writer.add_scalar(loss_name, loss_avg, epoch)
        log_writer.add_scalar(acc_name, acc_avg, epoch)


def report_lr(epoch, lr_name, lr, logger, log_writer):
    logger.info("Epoch[{}] learning-rate: {}".format(epoch, lr))
    if log_writer:
        log_writer.add_scalar(lr_name, lr, epoch)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_model(args, base_model, optimizer, scheduler, is_best, num_classes, class_names, epoch, acc1, logger):
    filepath = '{}-{}-{:04}.model'.format(args.prefix, args.model, epoch+1)
    savepath = os.path.join(args.model_dir, filepath)
    state = {
        'model': base_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'arch': args.model,
        'num_classes': num_classes,
        'class_names': class_names,
        'args': args,
        'epoch': epoch + 1,
        'acc1': float(acc1)
    }
    os.makedirs(args.model_dir, exist_ok=True)

    if not (args.save_best_only or args.save_best_and_last):
        torch.save(state, savepath)
        logger.info("=> Saved checkpoint to \"{}\"".format(savepath))

    if is_best:
        filepath = '{}-{}-best.model'.format(args.prefix, args.model)
        bestpath = os.path.join(args.model_dir, filepath)
        if args.save_best_only or args.save_best_and_last:
            torch.save(state, bestpath)
        else:
            shutil.copyfile(savepath, bestpath)
        logger.info("=> Saved checkpoint to \"{}\"".format(bestpath))

    if (args.epochs - 1 == epoch) and args.save_best_and_last:
        torch.save(state, savepath)
        logger.info("=> Saved checkpoint to \"{}\"".format(savepath))


def load_checkpoint(args, model_path):
    device = torch.device("cuda" if args.cuda else "cpu")
    print("=> loading saved checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint


def load_model_from_checkpoint(args, checkpoint, test_num_classes, test_class_names):
    device = torch.device("cuda" if args.cuda else "cpu")
    model_arch = checkpoint['arch']
    num_classes = checkpoint.get('num_classes', 0)
    if num_classes == 0:
        num_classes = test_num_classes
    base_model = make_model(model_arch, num_classes=num_classes, pretrained=False)
    base_model.load_state_dict(checkpoint['model'])
    class_names = checkpoint.get('class_names', [])
    if len(class_names) == 0:
        class_names = test_class_names

    if args.cuda:
        model = nn.DataParallel(base_model)
    else:
        model = base_model
    model.to(device)

    return model, num_classes, class_names


def check_args(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.mixup and args.ricap:
        warnings.warn('You can only one of the --mixup and --ricap can be activated.')
        sys.exit(1)

    if args.cutout and args.random_erasing:
        warnings.warn('You can only one of the --cutout and --random-erasing can be activated.')
        sys.exit(1)

    try:
        args.lr_step_epochs = [int(epoch) for epoch in args.lr_step_epochs.split(',')]
    except ValueError:
        warnings.warn('invalid --lr-step-epochs')
        sys.exit(1)

    try:
        args.random_resized_crop_scale = [float(scale) for scale in args.random_resized_crop_scale.split(',')]
        if len(args.random_resized_crop_scale) != 2:
            raise ValueError
    except ValueError:
        warnings.warn('invalid --random-resized-crop-scale')
        sys.exit(1)

    try:
        args.random_resized_crop_ratio = [float(ratio) for ratio in args.random_resized_crop_ratio.split(',')]
        if len(args.random_resized_crop_ratio) != 2:
            raise ValueError
    except ValueError:
        warnings.warn('invalid --random-resized-crop-ratio')
        sys.exit(1)

    if args.prefix == 'auto':
        args.prefix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    if args.workers is None:
        args.workers = max(1, int(0.8 * cpu_count()))
    elif args.workers == -1:
        args.workers = cpu_count()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.rgb_mean = [float(mean) for mean in args.rgb_mean.split(',')]
    args.rgb_std = [float(std) for std in args.rgb_std.split(',')]

    if args.model == 'pnasnet5large':
        scale_size = 352
        input_size = 331
    elif 'inception' in args.model:
        scale_size = 320
        input_size = 299
    elif 'xception' in args.model:
        scale_size = 320
        input_size = 299
    else:
        scale_size = 256
        input_size = 224

    if args.scale_size:
        scale_size = args.scale_size
    else:
        args.scale_size = scale_size
    if args.input_size:
        input_size = args.input_size
    else:
        args.input_size = input_size

    if not args.cutout:
        args.cutout_holes = None
        args.cutout_length = None

    if not args.random_erasing:
        args.random_erasing_p = None
        args.random_erasing_r1 = None
        args.random_erasing_r2 = None
        args.random_erasing_sh = None
        args.random_erasing_sl = None

    if not args.mixup:
        args.mixup_alpha = None

    if not args.ricap:
        args.ricap_beta = None
        args.ricap_with_line = False

    return args


def custom_six_crop(img, size):
    """Crop the given PIL Image into custom six crops.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center, full)
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    full = resize(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center, full)


def custom_seven_crop(img, size):
    """Crop the given PIL Image into custom seven crops.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center, semi_full, full)
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    shift_w = int(round(w - crop_w) / 4.)
    shift_h = int(round(h - crop_h) / 4.)

    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    semi_full = resize(img.crop((shift_w, shift_h, w - shift_w, h - shift_h)), (crop_h, crop_w))
    full = resize(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center, semi_full, full)


def custom_ten_crop(img, size):
    """Crop the given PIL Image into custom ten crops.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center, tl2, tr2, bl2, br2, full)
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    shift_w = int(round(w - crop_w) / 4.)
    shift_h = int(round(h - crop_h) / 4.)

    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    tl2 = img.crop((shift_w, shift_h, crop_w + shift_w, crop_h + shift_h))  # + +
    tr2 = img.crop((w - crop_w - shift_w, shift_h, w - shift_w, crop_h + shift_h))  # - +
    bl2 = img.crop((shift_w, h - crop_h - shift_h, crop_w + shift_w, h - shift_h))  # + -
    br2 = img.crop((w - crop_w - shift_w, h - crop_h - shift_h, w - shift_w, h - shift_h))  # - -
    full = resize(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center, tl2, tr2, bl2, br2, full)


def custom_twenty_crop(img, size, vertical_flip=False):
    r"""Crop the given PIL Image into custom twenty crops.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
       vertical_flip (bool): Use vertical flipping instead of horizontal
    Returns:
       tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_ten = custom_ten_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_ten = custom_ten_crop(img, size)
    return first_ten + second_ten


class CustomSixCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return custom_six_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CustomSevenCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return custom_seven_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CustomTenCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return custom_ten_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CustomTwentyCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return custom_twenty_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
