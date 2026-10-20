import os
import torch
import torch.distributed as dist


def setup():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1:
        local = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local)
        dist.init_process_group(backend='nccl')
        return True, world, int(os.environ['RANK']), local
    return False, 1, 0, 0


def is_main(rank):
    return rank == 0


def cleanup(enabled):
    if enabled and dist.is_initialized():
        dist.destroy_process_group()
