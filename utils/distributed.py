from collections import OrderedDict

import os
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch._utils import _flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors
from torch._utils import _take_tensors


def init_dist(launcher='pytorch', backend='nccl', **kwargs):
    if dist.is_initialized():
        return torch.cuda.current_device()
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(backend=backend, **kwargs)
    return gpu_id


def get_rank():
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_world_size():
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    return world_size


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


def is_master():
    """check if current process is the master"""
    return get_rank() == 0


@master_only
def master_only_print(*args):
    """master-only print"""
    print(*args)


def dist_reduce_tensor(tensor):
    """ Reduce to rank 0 """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if get_rank() == 0:
            tensor /= world_size
    return tensor


def dist_all_reduce_tensor(tensor):
    """ Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        tensor.div_(world_size)
    return tensor


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def allreduce_grads(model, coalesce=True, bucket_size_mb=-1):
    grads = [
        param.grad.data for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


class AllReduceDistributedDataParallel(nn.Module):

    def __init__(self, module, dim=0, broadcast_buffers=True,
                 bucket_cap_mb=25):
        super(AllReduceDistributedDataParallel, self).__init__()
        self.module = module
        self.dim = dim
        self.broadcast_buffers = broadcast_buffers

        self.broadcast_bucket_size = bucket_cap_mb * 1024 * 1024
        self._sync_params()

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, 0)
            for tensor, synced in zip(
                    tensors, _unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def _sync_params(self):
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states,
                                           self.broadcast_bucket_size)
        if self.broadcast_buffers:
            buffers = [b.data for b in self.module.buffers()]
            if len(buffers) > 0:
                self._dist_broadcast_coalesced(buffers,
                                               self.broadcast_bucket_size)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])
