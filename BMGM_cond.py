# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Generative Conditional Bridge Matching Diffusion Model on simple datasets

This script demostrates:
  - Creating the diffusion process, U-Net model, and optimizer.
  - Loading MNIST or CIFAR10 datasets.
  - Training the model  for N epochs saving the at the end the model.

Usage:
  For MNIST:
    python BMGM_cond.py --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 2e-3 --dataset mnist
  For CIFAR10:
    python BMGM_cond.py --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 1e-4 --dataset cifar10
"""

import os
import math
import argparse
import numpy as np
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils

# from torchmetrics.image.fid import FrechetInceptionDistance

import utils.unet as unet

import matplotlib.pyplot as plt


def match_last_dims(data, size):
    """
    Repeat a 1D tensor so that its last dimensions [1:] match `size[1:]`.
    Useful for working with batched data.
    """
    assert len(data.size()) == 1, "Data must be 1-dimensional (one value per batch)"
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))

class DiffusionProcess:
    def __init__(self, device=None, T=1.0, **kwargs):
        """
        process_type: 'VP' (variance preserving) or 'VE' (variance exploding)
        schedule: for VP, choose 'linear' or 'cosine'
        T: time horizon (used to sample t ~ Uniform(0,T)); the neural net always receives normalized time in [0,1]
        """
        self.device = 'cpu' if device is None else device
        self.T = T

    def loss_fn(self, model, x_start, y, model_kwargs=None, **kwargs):
        """
        Bridge Matching loss for the diffusion process.
        """
        x_1 = x_start.to(self.device)
        batch_size = x_start.shape[0]

        # Sample from the prior distribution (standard normal)
        x_0 = torch.randn_like(x_1)

        # Sample time t uniformly from [0,T]
        t = torch.rand(batch_size, device=self.device) * self.T
        t_comp = match_last_dims(t, x_start.shape)

        # Compute the mean and std for the bridge process
        mean_t = (1 - t_comp/self.T) * x_0 + (t_comp/self.T) * x_1
        std_t = 0.5 * torch.sqrt((t_comp * (self.T - t_comp)) / self.T)

        # Sample the new data point x_t
        x_t = mean_t + std_t * torch.randn_like(x_0)

        t_norm = t /self.T

        if model_kwargs is None:
            model_kwargs = {}

        # The model takes x_t and normalized time t_norm
        pred = model(x_t, t_norm, y, **model_kwargs)
        loss = F.mse_loss(pred, x_1)

        return {'loss': loss}

def create_unet(dataset):

    first_layer_embedding = False
    embedding_dim = 3 # MD4 needs a value for masks, so set of values is {0, 1, 2}
    output_dim = 1 # We only output a single probability value
    num_classes = 10
    if dataset == 'mnist':
        channels = 1
        out_channels = 1
        resnet_blocks = 2
        model_channels = 32
        attention_ds = (2,4)
        dropout = 0.0
    elif dataset == 'cifar10':
        channels = 3
        out_channels = 3
        resnet_blocks = 3
        model_channels = 128
        attention_ds = (8,16)
        dropout = 0.3

    model = unet.UNetModel(
            in_channels=channels,
            model_channels=model_channels,
            out_channels= out_channels,
            num_res_blocks=resnet_blocks,
            attention_resolutions= attention_ds,
            dropout= dropout,
            channel_mult= [1, 2, 2, 2], # divides image_size by two at each new item, except first one. [i] * model_channels
            dims = 2, # for images
            num_classes= num_classes,
            num_heads=4,
            num_heads_upsample=-1, # same as num_heads
            use_scale_shift_norm=True,
            first_layer_embedding=first_layer_embedding,
            embedding_dim= embedding_dim,
            output_dim = output_dim,
        )
    return model

def create_optimizer(model, lr=1e-3):
    """
    Returns an Adam optimizer for the model.
    """
    return optim.AdamW(model.parameters(), 
                                lr=lr, 
                                betas=(0.9, 0.999))

def get_device():
    """
    Returns the available device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    
def create_dataloader(dataset, batch_size):
    """
    Returns a DataLoader for the specified dataset.
    """
    if dataset == 'mnist':
        logging.info('Using MNIST dataset')
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.clamp(0, 1))
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset_obj = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)
    elif dataset =='cifar10':
        logging.info('Using CIFAR10 dataset')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2616])
        ])

        dataset_obj = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)

    return dataloader


def train(num_epochs, checkpoint_interval, batch_size, learning_rate, checkpoint_dir, dataset):
    device = get_device()
    logging.info(f'Using device: {device}')

    # Create objects
    diffusion_process = DiffusionProcess(device=device)
    model = create_unet(dataset).to(device) # the model depends on the dataset
    optimizer = create_optimizer(model, lr=learning_rate)
    dataloader = create_dataloader(dataset, batch_size)

    model.train()
    epoch_losses = []
    for epoch in (range(1, num_epochs + 1)):
        running_loss = 0.0
        for batch_idx, (data, y) in (enumerate(dataloader)):
            data = data.to(device) 
            y = y.to(device)
            optimizer.zero_grad()

            # Compute the training loss.
            losses_dict = diffusion_process.loss_fn(model, x_start=data, y=y.int())
            loss = losses_dict['loss']
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        logging.info(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

        # Save a checkpoint every checkpoint_interval epochs.
        if epoch % checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_losses': epoch_losses,
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

    logging.info("Training finished.")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("BMGM_uncond.txt"),       # saves to file
            logging.StreamHandler()                        # prints to stdout
        ]
    )
    parser = argparse.ArgumentParser(description="Conditional Bridge Matching Diffusion Training Script")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands: train")

    # Sub-parser for training.
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (N)")
    train_parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint interval (every C epochs)")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "cifar10"])
    train_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/conditional/BM/", help="Directory to save checkpoints")

    args = parser.parse_args()

    train(
            num_epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            dataset=args.dataset
        )
    
    parser.print_help()