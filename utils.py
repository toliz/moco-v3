import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / self.warmup_steps
        else:
            scale = (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps)/ (self.T_max - self.warmup_steps))) / 2
        
        return [self.eta_min + (base_lr - self.eta_min) * scale for base_lr in self.base_lrs]
    
    
class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
