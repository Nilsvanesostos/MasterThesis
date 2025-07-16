#!/usr/bin/env python
"""
Generative Conditional Score Matching Diffusion Model on simple datasets

This script demostrates:
  - Creating the diffusion process, U-Net model, and optimizer.
  - Loading MNIST or CIFAR10 datasets.
  - Training the model  for N epochs saving the at the end the model.

Usage:
  For MNIST:
    python SMGM.py train --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 2e-3 --dataset mnist --checkpoint_dir "./checkpoints/conditional/SM/mnist/" --schedule "linear"
  For CIFAR10:
    python SMGM.py train --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 1e-4 --dataset cifar10 --checkpoint_dir "./checkpoints/conditional/SM/mnist/" --schedule "cosine"
"""

import os
import math
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils

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
    def __init__(self, device=None, T=1.0, schedule='linear', **kwargs):
        """
        process_type: 'VP' (variance preserving) or 'VE' (variance exploding)
        schedule: for VP, choose 'linear' or 'cosine'
        T: time horizon (used to sample t ~ Uniform(0,T)); the neural net always receives normalized time in [0,1]
        """
        self.device = 'cpu' if device is None else device
        self.T = T
        self.schedule = schedule
        
        # For the linear noise schedule
        self.beta_min = kwargs.get('beta_min', 0.1)
        self.beta_max = kwargs.get('beta_max', 20.0)
        
        # For the cosine noise schedule
        self.s = kwargs.get('s', 0.008)
    
    
    def compute_alpha_bar(self, t_norm, schedule):
        """
        For linear: ᾱ(t) = exp( - [β_min t + 0.5 (β_max - β_min)t^2] )
        For cosine: ᾱ(t) = cos( ((t + s)/(1+s))*(π/2) )^2
        """
        if schedule == 'linear':
            integrated_beta = self.beta_min * t_norm + 0.5 * (self.beta_max - self.beta_min) * t_norm**2
            return torch.exp(-integrated_beta)
        elif schedule == 'cosine':
            return torch.cos((t_norm + self.s) / (1 + self.s) * (math.pi / 2))**2
        
    def compute_beta_bar(self, t_norm, schedule):
        """
        For linear: 
        For cosine:
        """
        if schedule == 'linear':
            beta_t = self.beta_min + t_norm * (self.beta_max - self.beta_min)
        elif schedule == 'cosine':
            beta_t = (math.pi / (self.T * (1 + self.s))) * torch.tan(((t_norm + self.s) / (1 + self.s)) * (math.pi / 2))
        return beta_t


    def score_fn(self, model, x, t, schedule):
        """
        Given the noise-predicting model, returns the score (i.e. ∇_x log p_t(x))
        at actual time t. Note that the model expects a normalized time (t/T).
        score = - (predicted noise) / sqrt(1 - ᾱ(t))

        """
        t_norm = t / self.T  # normalize to [0,1]
        alpha_bar = self.compute_alpha_bar(t_norm, schedule).view(-1, *([1] * (x.dim() - 1)))
        epsilon = model(x, t_norm)
        score = -epsilon / torch.sqrt(1 - alpha_bar)
        return score
    def forward(self, x_start, t, schedule):
        """
        Forward (diffusion) process: given a clean sample x_start and time t (in [0,T]),
        returns the noised version x_t.
        x_t = sqrt(ᾱ(t)) x_start + sqrt(1-ᾱ(t)) noise
        """
        t_norm = t / self.T  # normalize to [0,1]
        noise = torch.randn_like(x_start)
        alpha_bar = self.compute_alpha_bar(t_norm, schedule).view(-1, *([1] * (x_start.dim() - 1)))
        x_t = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def noise_loss(self, model, x_start, y, schedule, model_kwargs=None, **kwargs):
        """
        Training loss for the diffusion process.
        Samples t ~ Uniform(0, T), applies the forward process, and then
        computes the MSE loss between the network’s predicted noise and the true noise.
        """
        x_start = x_start.to(self.device)
        batch_size = x_start.shape[0]
        # Sample t uniformly from [0, T]
        t = torch.rand(batch_size, device=self.device) * self.T
        t_norm = t / self.T
        x_t, noise = self.forward(x_start, t_norm, schedule)
        
        if model_kwargs is None:
            model_kwargs = {}
        # The model takes x_t and normalized time t_norm
        predicted_noise = model(x_t, t_norm, y, **model_kwargs)
        loss = F.mse_loss(predicted_noise, noise)
        return {'loss': loss}

    def sample(self, model, n_samples, reverse_steps, schedule, dataset, interpolant, get_sample_history=False,  **kwargs):
        """
        SDE sampling using the Euler–Maruyama method to solve the reverse-time SDE:
          dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dẆ
        ODE sampling using the probability flow:
          dx = [f(x,t) - g(t)^2 * score(x,t) / 2] dt
        For VP:
          f(x,t) = -0.5 β(t)x  and  g(t) = sqrt(β(t))
          where β(t) is given either by a linear or cosine schedule.
        """
        if dataset == 'mnist':
            channels = 1
        elif dataset == 'cifar10':
            channels = 3
        else:
            raise ValueError("Unknown dataset")

        shape = (n_samples, channels, 32, 32)
        # Initialize x_T (the prior sample)
        xt = torch.randn(shape, device=self.device)
        all_images = []
        model.eval()
        with torch.inference_mode():
            # Create a time discretization from T to 0
            t_seq = torch.linspace(self.T, 0, reverse_steps + 1, device=self.device)
            for i in tqdm(range(reverse_steps)):
                t_current = t_seq[i]
                t_next = t_seq[i + 1]
                dt = t_next - t_current  # dt is negative (reverse time)
                # Create a batch of current time values for the update.
                t_batch = torch.full((shape[0],), t_current, device=self.device)
                t_norm_batch = t_batch / self.T

                beta_t = self.compute_beta_t(t_norm_batch, schedule).view(-1, *([1] * (xt.dim() - 1)))

                f = -0.5 * beta_t * xt
                g = torch.sqrt(beta_t)

                # Get the score (using the nn)
                score = self.score_fn(model, xt, t_batch)

                if interpolant == 'deterninistic':
                    xt = xt + (f - (g**2) * score / 2) * dt
                elif interpolant == 'stochastic':
                    z = torch.randn_like(xt)
                    xt = xt + (f - (g**2) * score) * dt + g * torch.sqrt(-dt) * z

                else:
                    raise("Unknown interpolant")
                
                if get_sample_history:
                            all_images.append(xt.clone())
        return xt if not get_sample_history else torch.stack(all_images)
    

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
        print('Using MNIST dataset')
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
        print('Using CIFAR10 dataset')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2616])
        ])

        dataset_obj = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dataloader


def train(num_epochs, checkpoint_interval, batch_size, learning_rate, checkpoint_dir, dataset, schedule):
    device = get_device()
    print('Using device:',device)

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
            losses_dict = diffusion_process.noise_loss(model, x_start=data, y=y.int(), schedule=schedule)
            loss = losses_dict['loss']
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

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
            print("Saved checkpoint to", checkpoint_path)

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Conditional Score Matching Diffusion Training Script")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands: train")

    # Sub-parser for training.
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (N)")
    train_parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint interval (every C epochs)")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "cifar10"])
    train_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/conditional/SM/", help="Directory to save checkpoints")
    train_parser.add_argument("--schedule", type=str, help="Noise schedule to use", choices=["linear", "cosine"])

    args = parser.parse_args()

    train(
            num_epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            dataset=args.dataset,
            schedule=args.schedule
        )
    
    parser.print_help()