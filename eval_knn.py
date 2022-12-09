import argparse
import torch
import torch.nn as nn
from moco import MoCo
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms


def extract_features(model: nn.Module, dataset: Dataset, args):
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            pin_memory=True, num_workers=args.num_workers)
    
    features, labels = [], []
    for batch_images, batch_labels in dataloader:
        batch_features = model(batch_images)
        
        features.append(batch_features)
        labels.append(batch_labels)
        
    features = nn.functional.normalize(torch.vstack(features), dim=1)
    labels = torch.cat(labels)
    
    return features, labels


def main(args):
    # Set-up model
    if args.pretrained_path is not None:
        moco = MoCo.load_from_checkpoint(args.pretrained_path)
    else:
        moco = MoCo()
    
    # Set-up training & evaluation data
    train_dataset = datasets.STL10(
        root=args.data_dir,
        split='train',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))
    test_dataset = datasets.STL10(
        root=args.data_dir,
        split='test',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))
    train_features, train_labels = extract_features(moco, train_dataset, args)
    test_features, test_labels = extract_features(moco, test_dataset, args)
    
    # Find k nearest neighbors
    similarity = torch.mm(test_features, train_features.T)
    _, indices = similarity.topk(args.k, largest=True, sorted=True)
    retrieved_neighbors = train_labels[indices]
    predictions, _ = torch.mode(retrieved_neighbors)

    # Calculate kNN accuracy
    acc = torch.mean((predictions == test_labels).float())
    
    print(f'k-NN accuracy: {acc}')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    parser.add_argument('--pretrained-path', default=None, type=str,
                        help='path to moco pretrained checkpoint')
    parser.add_argument('--data-dir', default='/scratch', type=str,
                        help='Root dir to store STL10')
    parser.add_argument('-k', default=10, type=int,
                        help='the number of neighbors in the algorithm')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='batch size to extract features')
    parser.add_argument('-j', '--num-workers', default=6, type=int,
                        help='number of data loading workers (default: 6)')
    
    args = parser.parse_args()
    
    main(args)
