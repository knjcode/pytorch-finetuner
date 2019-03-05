#!/usr/bin/env python
# coding: utf-8

import torch.optim as optim
import adabound


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
    elif args.optimizer == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=args.base_lr, weight_decay=args.wd, final_lr=args.final_lr)
    elif args.optimizer == 'amsbound':
        optimizer = adabound.AdaBound(model.parameters(), lr=args.base_lr, weight_decay=args.wd, final_lr=args.final_lr, amsbound=True)
    else:
        raise 'unknown optimizer'

    return optimizer
