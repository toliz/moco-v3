import argparse
import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from moco import MoCo
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from vit import ViTModule


def main(args):
    # Adapt base RL
    num_devices = max(torch.cuda.device_count(), 1)
    args.lr = args.lr * (args.batch_size * num_devices) / 256
    
    if args.seed:
        pl.seed_everything(args.seed)
    
    # Set-up training & evaluation data
    print('Downloading / Loading dataset...')
    train_dataset = datasets.STL10(
        root=args.data_dir,
        split='train',
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
    )
    val_dataset = datasets.STL10(
        root=args.data_dir,
        split='test',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True,
                                 num_workers=args.num_workers)
    
    # Set-up model
    print('Building ViT...')
    vit = ViTModule(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_epochs * len(train_dataloader) / num_devices,
        max_steps=args.max_epochs * len(train_dataloader) / num_devices, 
        finetune=args.pretrained_path is not None
    )
    if args.pretrained_path is not None:
        moco = MoCo.load_from_checkpoint(args.pretrained_path)
        vit.load_from_MoCo(moco)
    
    # Train
    logger = TensorBoardLogger(
        save_dir='tb_logs',
        name='train' if args.pretrained_path is None else 'finetune',
        default_hp_metric=False
    )
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        log_every_n_steps=10,
        accelerator='auto',
        devices=num_devices,
        strategy='ddp',
        sync_batchnorm=torch.cuda.device_count() > 1
    )
    trainer.fit(vit, train_dataloader, val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    # ViT args
    parser.add_argument('--pretrained-path', default=None, type=str,
                        help='path to moco pretrained checkpoint')

    # Dataset args
    parser.add_argument('--data-dir', default='/scratch', type=str,
                        help='Root dir to store STL10')
    
    # Trainer args
    parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                        help='number of epochs to warmup the learning rate')
    parser.add_argument('--max-epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 1024), this is the total '
                            'batch size of all GPUs on all nodes when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--num-workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--lr', '--learning-rate', default=1.5e-4, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                        metavar='W', help='weight decay (default: 0.1)',
                        dest='weight_decay')
    
    args = parser.parse_args()
    
    main(args)
