import argparse
import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from moco import MoCo
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils import ContrastiveTransformations


def main(args):
    # Adapt base RL
    num_devices = max(torch.cuda.device_count(), 1)
    args.lr = args.lr * (args.batch_size * num_devices) / 256
    
    if args.seed:
        pl.seed_everything(args.seed)
    
    # Set-up training data
    print('Downloading / Loading dataset...')
    contrast_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    dataset = datasets.STL10(root=args.data_dir, split='unlabeled', download=True,
                             transform=ContrastiveTransformations(contrast_transforms))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, pin_memory=True, num_workers=args.num_workers)
   
    
    # Set-up model
    print('Building MoCo-v3...')
    model = MoCo(
        args.encoder,
        args.out_dim,
        args.mlp_dim,
        args.tau,
        args.mu,
        args.lr,
        args.weight_decay,
        args.warmup_epochs * len(dataloader) / num_devices,
        args.max_epochs * len(dataloader) / num_devices,
    )
        
    # Train
    logger = TensorBoardLogger(save_dir='tb_logs', name='pre-train', default_hp_metric=False)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        log_every_n_steps=10,
        accelerator='auto',
        devices=num_devices,
        strategy='ddp',
        sync_batchnorm=torch.cuda.device_count() > 1
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoCo STL-10 Pre-Training')
    
    # MoCo-v3 args
    parser.add_argument('--encoder', default='vit_small',
                        help='encoder architecture: (default: vit_small)')
    parser.add_argument('--out-dim', default=256, type=int,
                        help='feature dimension (default: 256)')
    parser.add_argument('--mlp-dim', default=4096, type=int,
                        help='hidden dimension in MLPs (default: 4096)')
    parser.add_argument('--mu', default=0.99, type=float,
                        help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--tau', default=0.2, type=float,
                        help='softmax temperature (default: 0.2)')
    
    # Dataset args
    parser.add_argument('--data-dir', default='/scratch', type=str,
                        help='Root dir to store STL10')

    # Trainer args
    parser.add_argument('--max-epochs', default=15, type=int,
                        help='Total number of training epochs (default: 15)')
    parser.add_argument('--warmup-epochs', default=5, type=int,
                        help='Epochs to linearly warm-up the learning rate (default: 5)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size per GPU (default: 128)')
    parser.add_argument('-j', '--num-workers', default=6, type=int,
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--lr', '--learning-rate', default=1.5e-4, type=float,
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                        help='weight decay (default: 0.1)', dest='weight_decay')
    
    args = parser.parse_args()
    
    main(args)
