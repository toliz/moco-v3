import copy
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import vit


class MoCo(pl.LightningModule):
    def __init__(self, encoder='vit_small', out_dim=256, mlp_dim=4096, tau=0.2, mu=0.99, lr=1.5e-4, weight_decay=0.1, max_epochs=100):
        super(MoCo, self).__init__()
        
        self.save_hyperparameters()
        
        # build backbone
        encoder = vit.__dict__[encoder]()
        hidden_dim = encoder.head.weight.shape[1]
        
        # build modules
        self.encoder = copy.deepcopy(encoder)
        self.encoder.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim), nn.BatchNorm1d(out_dim, affine=False)
        )
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, mlp_dim), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim), nn.BatchNorm1d(out_dim, affine=False)
        )
        
        # stop gradient in momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
            
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=2e-2*self.hparams.lr)
        
        return [optimizer], [scheduler]
    
    def forward(self, x, momentum=False):
        if momentum:
            with torch.no_grad():
                x = self.momentum_encoder(x)
        else:
            x = self.encoder(x)
            x = self.predictor(x)
        
        return x
    
    @torch.no_grad()
    def update_momentum_encoder(self):
        for param, momentum_param in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            momentum_param = momentum_param * self.hparams.mu + param * (1-self.hparams.mu)
    
    def contrastive_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        logits = q @ k.T
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = nn.functional.cross_entropy(logits / self.hparams.tau, labels)
        
        return 2 * self.hparams.tau * loss
        
    def training_step(self, batch, batch_idx):
        self.update_momentum_encoder()
        
        # forward pass
        (imgs_1, imgs_2), _ = batch
        q1, q2 = self(imgs_1), self(imgs_2)                                 # queries: [N, C] each
        k1, k2 = self(imgs_1, momentum=True), self(imgs_2, momentum=True)   # keys: [N, C] each
        
        # calculate MoCo contrastive loss
        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        self.log("MoCo-v3 loss", loss)
        
        return loss
        