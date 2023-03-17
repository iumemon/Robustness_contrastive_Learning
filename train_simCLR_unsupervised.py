import argparse
import torch
import torchvision
import utils
import simclr
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import os
from skimage import io
from os import walk
from dataloader_contrastive import TrainDataset
# making a command line interface
parser = argparse.ArgumentParser(
    description="This is the command line interface for the SimCLR framework for self-supervised learning. Below are the arguments which are required to run this program.")

parser.add_argument('datapath', type=str,
                    help="Path to the data root folder which contains train and test folders")

parser.add_argument('respath', type=str,
                    help="Path to the results directory where the saved model and evaluation graphs would be stored. ")

parser.add_argument('-bs', '--batch_size', default=64, type=int,
                    help="The batch size for self-supervised training")

parser.add_argument('-n', '--epochs', default=100, type=int,
                    help="Number of epochs for self supervised training")

parser.add_argument('-nw', '--num_workers', default=2,
                    type=int, help="The number of workers for loading data")

parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-resume', action='store_true')
parser.add_argument('--multiple_gpus', action='store_true')

random_transform = transforms.RandomApply(transforms=[
    transforms.RandomRotation(45, fill=255),
    transforms.RandomRotation(30, fill=255),
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0, fill=255)], p=1)


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    print("Training on: ",args.device)
    model = utils.model.get_model(args)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=1e-4
    )

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        TrainDataset(args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers = 4
    )

    loss_fn = utils.ntxent.loss_function
    if(args.resume):
        print(f"Resuming Training for: {args.epochs} epoch")
        checkpoint = torch.load("results/model/model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
    simclrobj = simclr.SimCLR(model, optimizer, dataloaders, loss_fn)
    simclrobj.train(args, num_epochs=args.epochs)
