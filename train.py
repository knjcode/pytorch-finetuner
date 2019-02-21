#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import sys
import time

import logzero
import numpy as np
import tensorboardX
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

from cnn_finetune import make_model
from PIL import ImageFile
from torchvision.utils import save_image
from logzero import logger

from util.dataloader import get_dataloader, CutoutForBatchImages, RandomErasingForBatchImages
from util.functions import check_args, get_lr, print_batch, report, report_lr, save_model, accuracy, Metric
from util.optimizer import get_optimizer
from util.scheduler import get_cosine_annealing_lr_scheduler, get_multi_step_lr_scheduler, get_reduce_lr_on_plateau_scheduler

import signal
import warnings

signal.signal(signal.SIGINT, signal.default_int_handler)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# argparse
parser = argparse.ArgumentParser(description='train')
parser.add_argument('data', metavar='DIR', help='path to dataset (train and validation)')

# model architecture
parser.add_argument('--model', '-m', metavar='ARCH', default='resnet18',
                    help='specify model architecture (default: resnet18)')
parser.add_argument('--from-scratch', dest='scratch', action='store_true',
                    help='do not use pre-trained weights (default: False)')

# epochs, batch sixe, etc
parser.add_argument('--epochs', type=int, default=30, help='number of total epochs to run (default: 30)')
parser.add_argument('--batch-size', '-b', type=int, default=128, help='the batch size (default: 128)')
parser.add_argument('--val-batch-size', type=int, default=256, help='the validation batch size (default: 256)')
parser.add_argument('-j', '--workers', type=int, default=None,
                    help='number of data loading workers (default: 80%% of the number of cores)')
parser.add_argument('--prefix', default='auto',
                    help="prefix of model and logs (default: auto)")
parser.add_argument('--log-dir', default='logs',
                    help='log directory (default: logs)')
parser.add_argument('--model-dir', default='model',
                    help='model saving dir (default: model)')
parser.add_argument('--resume', default=None, type=str, metavar='MODEL',
                    help='path to saved model (default: None)')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (default: 0)')
parser.add_argument('--disp-batches', type=int, default=0,
                    help='show progress for every n batches (default: auto)')
parser.add_argument('--save-best-only', action='store_true', default=False,
                    help='save only the latest best model according to the validation accuracy (default: False)')
parser.add_argument('--save-best-and-last', action='store_true', default=False,
                    help='save last and latest best model according to the validation accuracy (default: False)')

# optimizer, lr, etc
parser.add_argument('--base-lr', type=float, default=0.001,
                    help='initial learning rate (default: 0.001)')
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help='the ratio to reduce lr on each step (default: 0.1)')
parser.add_argument('--lr-step-epochs', type=str, default='10,20',
                    help='the epochs to reduce the lr (default: 10,20)')
parser.add_argument('--lr-patience', type=int, default=None,
                    help='enable ReduceLROnPlateau lr scheduler with specified patience (default: None)')
parser.add_argument('--cosine-annealing-t-max', type=int, default=None,
                    help='enable CosineAnnealinigLR scheduler with specified T_max (default: None)')
parser.add_argument('--cosine-annealing-mult', type=int, default=2,
                    help='T_mult of CosineAnnealingLR scheduler')
parser.add_argument('--cosine-annealing-eta-min', type=float, default=1e-05,
                    help='Minimum learning rate of CosineannealingLR scheduler')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the optimizer type (default: sgd)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-04,
                    help='weight decay (default: 1e-04)')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs (default: 5)')

# data preprocess and augmentation settings
parser.add_argument('--scale-size', type=int, default=None,
                    help='scale size (default: auto)')
parser.add_argument('--input-size', type=int, default=None,
                    help='input size (default: auto)')
parser.add_argument('--rgb-mean', type=str, default='0,0,0',
                    help='RGB mean (default: 0,0,0)')
parser.add_argument('--rgb-std', type=str, default='1,1,1',
                    help='RGB std (default: 1,1,1)')
parser.add_argument('--random-resized-crop-scale', type=str, default='0.08,1.0',
                    help='range of size of the origin size cropped (default: 0.08,1.0)')
parser.add_argument('--random-resized-crop-ratio', type=str, default='0.75,1.3333333333333333',
                    help='range of aspect ratio of the origin aspect ratio cropped (defaullt: 0.75,1.3333333333333333)')
parser.add_argument('--random-horizontal-flip', type=float, default=0.5,
                    help='probability of the image being flipped (default: 0.5)')
parser.add_argument('--random-vertical-flip', type=float, default=0.0,
                    help='probability of the image being flipped (default: 0.0)')
parser.add_argument('--jitter-brightness', type=float, default=0.10,
                    help='jitter brightness of data augmentation (default: 0.10)')
parser.add_argument('--jitter-contrast', type=float, default=0.10,
                    help='jitter contrast of data augmentation (default: 0.10)')
parser.add_argument('--jitter-saturation', type=float, default=0.10,
                    help='jitter saturation of data augmentation (default: 0.10)')
parser.add_argument('--jitter-hue', type=float, default=0.05,
                    help='jitter hue of data augmentation (default: 0.05)')
parser.add_argument('--random-rotate-degree', type=float, default=3.0,
                    help='rotate degree of data augmentation (default: 3.0)')

parser.add_argument('--image-dump', action='store_true', default=False,
                    help='dump batch images and exit (default: False)')
parser.add_argument('--calc-rgb-mean-and-std', action='store_true', default=False,
                    help='calculate rgb mean and std of train images and exit (default: False)')

# misc
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. (default: None)')
parser.add_argument('--warm_restart_next', type=int, default=None,
                    help='next warm restart epoch (default: None')
parser.add_argument('--warm_restart_current', type=int, default=None,
                    help='current warm restart epoch (default: None)')

# cutout
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout (default: False)')
parser.add_argument('--cutout-holes', type=int, default=1,
                    help='number of holes to cut out from image (default: 1)')
parser.add_argument('--cutout-length', type=int, default=64,
                    help='length of the holes (default: 64)')

# random erasing
parser.add_argument('--random-erasing', action='store_true', default=False,
                    help='apply random erasing (default: False)')
parser.add_argument('--random-erasing-p', type=float, default=0.5,
                    help='random erasing p (default: 0.5)')
parser.add_argument('--random-erasing-sl', type=float, default=0.02,
                    help='random erasing sl (default: 0.02)')
parser.add_argument('--random-erasing-sh', type=float, default=0.4,
                    help='random erasing sh (default: 0.4)')
parser.add_argument('--random-erasing-r1', type=float, default=0.3,
                    help='random erasing r1 (default: 0.3)')
parser.add_argument('--random-erasing-r2', type=float, default=1/0.3,
                    help='random erasing r2 (default: 3.3333333333333335)')

# mixup
parser.add_argument('--mixup', action='store_true', default=False,
                    help='apply mixup (default: Falsse)')
parser.add_argument('--mixup-alpha', type=float, default=0.2,
                    help='mixup alpha (default: 0.2)')

# ricap
parser.add_argument('--ricap', action='store_true', default=False,
                    help='apply RICAP (default: False)')
parser.add_argument('--ricap-beta', type=float, default=0.3,
                    help='RICAP beta (default: 0.3)')
parser.add_argument('--ricap-with-line', action='store_true', default=False,
                    help='RICAP with boundary line (default: False)')


best_acc1 = 0


def main():
    global args, best_acc1
    args = parser.parse_args()

    args = check_args(args)

    formatter = logging.Formatter('%(message)s')
    logzero.formatter(formatter)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    log_filename = "{}-train.log".format(args.prefix)
    logzero.logfile(os.path.join(args.log_dir, log_filename))

    # calc rgb_mean and rgb_std
    if args.calc_rgb_mean_and_std:
        calc_rgb_mean_and_std(args, logger)

    # setup dataset
    train_loader, train_num_classes, train_class_names, valid_loader, _valid_num_classes, _valid_class_names \
        = get_dataloader(args, args.scale_size, args.input_size)

    if args.disp_batches == 0:
        target = len(train_loader) // 10
        args.disp_batches = target - target % 5
    if args.disp_batches < 5:
        args.disp_batches = 1

    logger.info('Running script with args: {}'.format(str(args)))
    logger.info("scale_size: {}  input_size: {}".format(args.scale_size, args.input_size))
    logger.info("rgb_mean: {}".format(args.rgb_mean))
    logger.info("rgb_std: {}".format(args.rgb_std))
    logger.info("number of train dataset: {}".format(len(train_loader.dataset)))
    logger.info("number of validation dataset: {}".format(len(valid_loader.dataset)))
    logger.info("number of classes: {}".format(len(train_class_names)))

    if args.mixup:
        logger.info("Using mixup: alpha:{}".format(args.mixup_alpha))
    if args.ricap:
        logger.info("Using RICAP: beta:{}".format(args.ricap_beta))
    if args.cutout:
        logger.info("Using cutout: holes:{} length:{}".format(args.cutout_holes, args.cutout_length))
    if args.random_erasing:
        logger.info("Using Random Erasing: p:{} s_l:{} s_h:{} r1:{} r2:{}".format(
            args.random_erasing_p, args.random_erasing_sl, args.random_erasing_sh,
            args.random_erasing_r1, args.random_erasing_r2))

    device = torch.device("cuda" if args.cuda else "cpu")

    # create  model
    if args.resume:
        # resume from a checkpoint
        if os.path.isfile(args.resume):
            logger.info("=> loading saved checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.model = checkpoint['arch']
            base_model = make_model(args.model,
                                    num_classes=train_num_classes,
                                    pretrained=False,
                                    input_size=(args.input_size, args.input_size))
            base_model.load_state_dict(checkpoint['model'])
            args.start_epoch = checkpoint['epoch']
            best_acc1 = float(checkpoint['acc1'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.error("=> no checkpoint found at '{}'".format(args.resume))
            sys.exit(1)
    else:
        if args.scratch:
            # train from scratch
            logger.info("=> creating model '{}' (train from scratch)".format(args.model))
            base_model = make_model(args.model,
                                    num_classes=train_num_classes,
                                    pretrained=False,
                                    input_size=(args.input_size, args.input_size))
        else:
            # fine-tuning
            logger.info("=> using pre-trained model '{}'".format(args.model))
            base_model = make_model(args.model,
                                    num_classes=train_num_classes,
                                    pretrained=True,
                                    input_size=(args.input_size, args.input_size))

    if args.cuda:
        logger.info("=> using GPU")
        model = nn.DataParallel(base_model)
        model.to(device)
    else:
        logger.info("=> using CPU")
        model = base_model

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    logger.info('=> using optimizer: {}'.format(args.optimizer))
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> restore optimizer state from checkpoint")

    # create scheduler
    if args.lr_patience:
        scheduler = get_reduce_lr_on_plateau_scheduler(args, optimizer)
        logger.info("=> using ReduceLROnPlateau scheduler")
    elif args.cosine_annealing_t_max:
        scheduler = get_cosine_annealing_lr_scheduler(args, optimizer, args.cosine_annealing_t_max, len(train_loader))
        logger.info("=> using CosineAnnealingLR scheduler")
    else:
        scheduler = get_multi_step_lr_scheduler(args, optimizer, args.lr_step_epochs, args.lr_factor)
        logger.info("=> using MultiStepLR scheduler")
    if args.resume:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("=> restore lr scheduler state from checkpoint")

    logger.info("=> model and logs prefix: {}".format(args.prefix))
    logger.info("=> log dir: {}".format(args.log_dir))
    logger.info("=> model dir: {}".format(args.model_dir))
    tensorboradX_log_dir = os.path.join(args.log_dir, "{}-tensorboardX".format(args.prefix))
    log_writer = tensorboardX.SummaryWriter(tensorboradX_log_dir)
    logger.info("=> tensorboardX log dir: {}".format(tensorboradX_log_dir))

    if args.cuda:
        cudnn.benchmark = True

    if args.lr_patience:  # ReduceLROnPlateau
        scheduler.step(float('inf'))
    elif not args.cosine_annealing_t_max:  # MultiStepLR
        scheduler.step()

    # for CosineAnnealingLR
    if args.resume:
        args.warm_restart_next = checkpoint['args'].warm_restart_next
        args.warm_restart_current = checkpoint['args'].warm_restart_current
    else:
        if args.cosine_annealing_t_max:  # CosineAnnealingLR
            args.warm_restart_next = args.cosine_annealing_t_max + args.warmup_epochs
            args.warm_restart_current = args.warmup_epochs

    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()

        # CosineAnnealingLR warm restart
        if args.cosine_annealing_t_max and (epoch % args.warm_restart_next == 0) and epoch != 0:
            current_span = args.warm_restart_next - args.warm_restart_current
            next_span = current_span * args.cosine_annealing_mult
            args.warm_restart_current = args.warm_restart_next
            args.warm_restart_next = args.warm_restart_next + next_span
            scheduler = get_cosine_annealing_lr_scheduler(args, optimizer, next_span, len(train_loader))

        if args.mixup:
            train(args, 'mixup', train_loader, model, device, criterion, optimizer, scheduler, epoch, logger, log_writer)
        elif args.ricap:
            train(args, 'ricap', train_loader, model, device, criterion, optimizer, scheduler, epoch, logger, log_writer)
        else:
            train(args, 'normal', train_loader, model, device, criterion, optimizer, scheduler, epoch, logger, log_writer)

        report_lr(epoch, 'x_learning_rate', get_lr(optimizer), logger, log_writer)

        acc1 = valid(args, valid_loader, model, device, criterion, optimizer, scheduler, epoch, logger, log_writer)

        elapsed_time = time.time() - start
        logger.info("Epoch[{}] Time cost: {} [sec]".format(epoch, elapsed_time))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_model(args, base_model, optimizer, scheduler, is_best, train_num_classes, train_class_names, epoch, acc1, logger)


def train(args, train_mode, train_loader, model, device, criterion, optimizer, scheduler, epoch, logger, log_writer):
    total_size = 0
    data_size = len(train_loader.dataset)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    model.train()

    start = time.time()

    alpha = args.mixup_alpha
    beta = args.ricap_beta

    if train_mode in ['mixup', 'ricap']:
        if args.cutout:
            batch_cutout = CutoutForBatchImages(n_holes=args.cutout_holes, length=args.cutout_length)
        if args.random_erasing:
            batch_random_erasing = RandomErasingForBatchImages(p=args.random_erasing_p,
                                                            sl=args.random_erasing_sl,
                                                            sh=args.random_erasing_sh,
                                                            r1=args.random_erasing_r1,
                                                            r2=args.random_erasing_r2)

    for batch_idx, (data, target, _paths) in enumerate(train_loader):
        adjust_learning_rate(args, epoch, batch_idx, train_loader, optimizer, scheduler, logger)

        if train_mode is 'mixup':
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
            batch_size = data.size()[0]
            if args.cuda:
                index = torch.randperm(batch_size).cuda()
            else:
                index = torch.randperm(batch_size)
            mixed_data = lam * data + (1 - lam) * data[index, :]
            target_a, target_b = target, target[index]

            if args.cutout:
                mixed_data = batch_cutout(mixed_data)
            if args.random_erasing:
                mixed_data = batch_random_erasing(mixed_data)

        elif train_mode is 'ricap':
            I_x, I_y = data.size()[2:]
            w = int(np.round(I_x * np.random.beta(beta, beta)))
            h = int(np.round(I_y * np.random.beta(beta, beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]
            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                index = torch.randperm(data.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = data[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                if args.cuda:
                    c_[k] = target[index].cuda(non_blocking=True)
                else:
                    c_[k] = target[index]
                W_[k] = w_[k] * h_[k] / (I_x * I_y)
            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                torch.cat((cropped_images[2], cropped_images[3]), 2)), 3)
            # draw lines
            if args.ricap_with_line:
                patched_images[:, :, w-1:w+1, :] = 0.
                patched_images[:, :, :, h-1:h+1] = 0.

            if args.cutout:
                patched_images = batch_cutout(patched_images)
            if args.random_erasing:
                patched_images = batch_random_erasing(patched_images)

        # normal train mode applies Cutout or Random Erasing inside dataloader

        if args.image_dump:
            if train_mode is 'mixup':
                save_image(mixed_data, './samples.jpg')
            elif train_mode is 'ricap':
                save_image(patched_images, './samples.jpg')
            else:
                save_image(data, './samples.jpg')
            logger.info("image saved! at ./samples.jpg")
            sys.exit(0)

        if args.cuda:
            if train_mode is 'mixup':
                mixed_data = mixed_data.cuda(non_blocking=True)
                target_a = target_a.cuda(non_blocking=True)
                target_b = target_b.cuda(non_blocking=True)
            elif train_mode is 'ricap':
                patched_images = patched_images.cuda(non_blocking=True)
            else:  # vanila train
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        # output and loss
        if train_mode is 'mixup':
            output = model(mixed_data)
            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
        elif train_mode is 'ricap':
            output = model(patched_images)
            loss = sum(W_[k] * criterion(output, c_[k]) for k in range(4))
        else:  # vanila train
            output = model(data)
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if train_mode is 'mixup':
            train_accuracy.update(lam * accuracy(output, target_a) + (1 - lam) * accuracy(output, target_b))
        elif train_mode is 'ricap':
            train_accuracy.update(sum([W_[k] * accuracy(output, c_[k]) for k in range(4)]))
        else:  # vanila train
            train_accuracy.update(accuracy(output, target))

        train_loss.update(loss)

        total_size += data.size(0)
        if (batch_idx + 1) % args.disp_batches == 0:
            prop = 100. * (batch_idx+1) / len(train_loader)
            elapsed_time = time.time() - start
            speed = total_size / elapsed_time
            print_batch(batch_idx+1, epoch, total_size, data_size, prop, speed, train_accuracy.avg, train_loss.avg, logger)

    report(epoch, 'Train', 'train/loss', train_loss.avg, 'train/accuracy', train_accuracy.avg, logger, log_writer)


def valid(args, valid_loader, model, device, criterion, optimizer, scheduler, epoch, logger, log_writer):
    valid_loss = Metric('valid_loss')
    valid_accuracy = Metric('valid_accuracy')
    model.eval()

    with torch.no_grad():
        for (data, target, _paths) in valid_loader:
            if args.cuda:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = model(data)
            loss = criterion(output, target)

            valid_accuracy.update(accuracy(output, target))
            valid_loss.update(loss)

        report(epoch, 'Validation', 'val/loss', valid_loss.avg, 'val/accuracy', valid_accuracy.avg, logger, log_writer)

        if args.lr_patience:  # ReduceLROnPlateau
            scheduler.step(valid_loss.avg)
        elif not args.cosine_annealing_t_max:  # MultiStepLR
            scheduler.step()

    return valid_accuracy.avg


def get_warmup_lr_adj(args, epoch, batch_idx, train_loader, optimizer, logger):
    lr_adj = 1.
    if epoch < args.warmup_epochs:
        epoch = epoch * len(train_loader)
        epoch += float(batch_idx + 1)
        lr_adj = epoch / (len(train_loader) * (args.warmup_epochs + 1))
    return lr_adj


def adjust_learning_rate(args, epoch, batch_idx, train_loader, optimizer, scheduler, logger):
    lr_adj = 1.
    if epoch < args.warmup_epochs:
        lr_adj = get_warmup_lr_adj(args, epoch, batch_idx, train_loader, optimizer, logger)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.base_lr * lr_adj
    else:
        if args.cosine_annealing_t_max:
            scheduler.step()


def calc_rgb_mean_and_std(args, logger):
    from util.dataloader import get_image_datasets_for_rgb_mean_and_std
    from util.functions import IncrementalVariance
    from tqdm import tqdm

    image_datasets = get_image_datasets_for_rgb_mean_and_std(args, args.scale_size, args.input_size)
    logger.info("=> Calculate rgb mean and std (dir: {}  images: {}  batch-size: {})".format(args.data, len(image_datasets), args.batch_size))

    if args.batch_size < len(image_datasets):
        logger.info("To calculate more accurate values, please specify as large a batch size as possible.")

    kwargs = {'num_workers': args.workers}
    train_loader = torch.utils.data.DataLoader(
        image_datasets, batch_size=args.batch_size, shuffle=False, **kwargs)

    iv = IncrementalVariance()
    processed = 0
    with tqdm(total=len(train_loader), desc="Calc rgb mean/std") as t:
        for data, _target in train_loader:
            numpy_images = data.numpy()
            batch_mean = np.mean(numpy_images, axis=(0, 2, 3))
            batch_var = np.var(numpy_images, axis=(0, 2, 3))
            iv.update(batch_mean, len(numpy_images), batch_var)
            processed += len(numpy_images)
            t.update(1)

    logger.info("=> processed: {} images".format(processed))
    logger.info("=> calculated rgb mean: {}".format(iv.average))
    logger.info("=> calculated rgb std: {}".format(iv.std))

    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    rgb_mean_option = np.array2string(iv.average, separator=',').replace('[', '').replace(']', '')
    rgb_std_option = np.array2string(iv.std, separator=',').replace('[', '').replace(']', '')
    logger.info("Please use following command options when train and test:")
    logger.info(" --rgb-mean {} --rgb-std {}".format(rgb_mean_option, rgb_std_option))

    sys.exit(0)


if __name__ == '__main__':
    main()
