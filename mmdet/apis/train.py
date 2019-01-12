from __future__ import division

from collections import OrderedDict

import logging
import torch
from torch._six import inf
from mmcv.runner import DistSamplerSeedHook
from mmcv.runner.hooks import (Hook, LrUpdaterHook, CheckpointHook, IterTimerHook, OptimizerHook, lr_updater)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.core import (DistOptimizerHook, CocoDistEvalRecallHook,
                        CocoDistEvalmAPHook)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from .env import get_root_logger
import os

from mmcv.runner import Runner as mmcvRunner
# from torch.nn.utils import clip_grad


import mmdet.flags as FLAGS
import mmdet
from random import shuffle


def set_width_mult(m, width_mult):
     if hasattr(m, 'width_mult'):
         m.width_mult = width_mult
         if hasattr(m, 'onehot'):
             m.onehot[:, :, :, :] = 0.
             channels = m.num_features_list[
                 FLAGS.width_mult_list.index(m.width_mult)]
             m.onehot[:, :channels, :, :] = 1.
         elif hasattr(m, 'in_channels_list'):
             m.current_in_channels = m.in_channels_list[
                 FLAGS.width_mult_list.index(m.width_mult)]
             m.current_out_channels = m.out_channels_list[
                 FLAGS.width_mult_list.index(m.width_mult)]
             m.current_groups = m.groups_list[
                 FLAGS.width_mult_list.index(m.width_mult)]
         else:
             pass

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    printall = []
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            if torch.distributed.get_rank() == 0:
                printall.append('{}: {}'.format(list(p.size()), param_norm))
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        # if torch.distributed.get_rank() == 0:
            # print('Total norm: {}'.format(total_norm))
            # print(', '.join(printall))
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

def clip_grads(params, width_mult=1.0):
    clip_grad_norm_(
        filter(lambda p: p.requires_grad, params), max_norm=35, norm_type=2)

class Runner(mmcvRunner):
    def __init__(self, model, batch_processor, optimizer=None, work_dir=None, log_level=logging.INFO):
        super(Runner, self).__init__(model, batch_processor, optimizer, work_dir, log_level)
        self.width_mult_list = FLAGS.width_mult_list.copy()

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')

            # slimmable
            self.optimizer.zero_grad()
            for width_mult in self.width_mult_list:
                FLAGS.width_mult_current = width_mult
                self.model.apply(lambda m: set_width_mult(m, width_mult=width_mult))
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=True, **kwargs)
                if not isinstance(outputs, dict):
                    raise TypeError('batch_processor() must return a dict')
                if 'log_vars' in outputs and width_mult == max(FLAGS.width_mult_list):
                    self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                self.outputs = outputs
                self.outputs['loss'].backward()
            # optimize
            mmdet.core.utils.dist_utils.allreduce_grads(self.model, True, -1)
            clip_grads(self.model.parameters(), width_mult)
            self.optimizer.step()

            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.
        Default hooks include:
        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        # self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        if isinstance(model.module, RPN):
            runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
        elif cfg.data.val.type == 'CocoDataset':
            runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))

    if os.path.exists(os.path.join(cfg.work_dir, 'latest.pth')):
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if os.path.exists(os.path.join(cfg.work_dir, 'latest.pth')):
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if cfg.resume_from:
        print('[Debug] Resume from: {}'.format(cfg.resume_from))
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
