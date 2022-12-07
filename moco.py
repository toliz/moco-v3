import copy
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import vit

from utils import CosineAnnealingWithWarmupLR


class MoCo(pl.LightningModule):
    def __init__(self, encoder='vit_small', out_dim=256, mlp_dim=4096, tau=0.2, mu=0.99, lr=1.5e-4, weight_decay=0.1, warmup_steps=1, max_steps=10):
        super(MoCo, self).__init__()
        
        self.save_hyperparameters()
        
        # build backbone
        encoder = vit.__dict__[encoder]()
        hidden_dim = encoder.head.weight.shape[1]
        
        # build modules
        self.encoder = copy.deepcopy(encoder)
        self.encoder.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim, bias=False), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim, bias=False), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, affine=False)
        )
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, mlp_dim, bias=False), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, affine=False)
        )
        
        # stop gradient in momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
            
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingWithWarmupLR(optimizer, self.hparams.warmup_steps, self.hparams.max_steps)
                
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def forward(self, x, momentum=False):
        if momentum:
            with torch.no_grad():
                x = self.momentum_encoder(x)
        else:
            x = self.encoder(x)
            x = self.predictor(x)
        
        return x
    
    def contrastive_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        k = concat_all_gather(k)
        
        logits = q @ k.T
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = nn.functional.cross_entropy(logits / self.hparams.tau, labels)
        
        return 2 * self.hparams.tau * loss
            
    def on_train_batch_start(self, batch, batch_idx):
        # Update mu with a cosine schedule
        current_step = self.current_epoch * self.trainer.num_training_batches + batch_idx
        mu = (1 - (1 + math.cos(math.pi * current_step / self.hparams.max_steps)) / 2) * (1-self.hparams.mu) + self.hparams.mu
        
        # Update momentum encoder
        with torch.no_grad():
            for param, momentum_param in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                momentum_param = momentum_param * mu + param * (1 - mu)
                
        self.log('mu', mu)
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'])
        
    def training_step(self, batch, batch_idx):
        # forward pass
        (imgs_1, imgs_2), _ = batch
        q1, q2 = self(imgs_1), self(imgs_2)                                 # queries: [batch_size, out_dim] each
        k1, k2 = self(imgs_1, momentum=True), self(imgs_2, momentum=True)   # keys: [batch_size, out_dim] each
        
        # calculate MoCo contrastive loss
        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        self.log('MoCo-v3 loss', loss)
        
        return loss


# utils
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
