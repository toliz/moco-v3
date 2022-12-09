# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from functools import partial, reduce
from operator import mul
from utils import CosineAnnealingWithWarmupLR

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed


__all__ = [
    'vit_small', 
    'vit_base',
]


class ViT(VisionTransformer):
    def __init__(self, stop_grad_conv1=True, **kwargs):
        super().__init__(img_size=96, **kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False
        
        
class ViTModule(ViT, pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-4, weight_decay=0.1, warmup_steps=1, max_steps=10, finetune=False):
        super(ViTModule, self).__init__(patch_size=8, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.save_hyperparameters()
        
        # Replace head with the correct number of classes
        self.head = nn.Linear(self.head.in_features, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()
        
        # When finetuning, freeze everything but the classification head
        if finetune:
            for name, param in self.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False
        
    def configure_optimizers(self):
        if self.hparams.finetune:
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(self.hparams.max_steps*0.3), int(self.hparams.max_steps*0.6)],
                gamma=0.1
            )
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            scheduler = CosineAnnealingWithWarmupLR(optimizer, self.hparams.warmup_steps, self.hparams.max_steps)    
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def load_from_MoCo(self, moco):
        state_dict = {k: v for (k, v) in moco.encoder.state_dict().items() if not k.startswith('head')}
        msg = self.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    
    def _calculate_loss(self, batch, mode='train'):
        imgs, labels = batch
        preds = self(imgs)
        loss = nn.functional.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'ViT {mode} loss', loss)
        self.log(f'ViT {mode} acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')


def vit_small(**kwargs):
    model = ViT(
        patch_size=8, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_base(**kwargs):
    model = ViT(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
