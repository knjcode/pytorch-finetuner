#!/usr/bin/env python
# coding: utf-8

import torch.optim as optim


def get_optimizer(args, model):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optimizer == 'nag':
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.wd)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.wd, amsgrad=True)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        raise 'unknown optimizer'

    return optimizer
