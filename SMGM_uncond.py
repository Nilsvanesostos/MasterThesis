# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Generative Unconditional Score Matching Diffusion Model on simple datasets

This script demostrates:
  - Creating the diffusion process, U-Net model, and optimizer.
  - Loading MNIST or CIFAR10 datasets.
  - Training the model  for N epochs saving the at the end the model.
  - Generation of images both with deterministic and stochastic interpolants.
  - Computation of the FID score of a generated dataset.

Usage:
  For training:
    For MNIST:
      python SMGM_uncond.py train --epochs 500 --checkpoint_interval 50 --batch_size 64 --learning_rate 2e-3 --dataset "mnist" --checkpoint_dir "./checkpoints/unconditional/SM/mnist/" --schedule "linear"
    For CIFAR10:
      python SMGM_uncond.py train --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 1e-4 --dataset "cifar10" --checkpoint_dir "./checkpoints/unconditional/SM/cifar10/" --schedule "cosine"
  For generation:   # checkpoint_path, n_samples, reverse_steps, schedule, interpolant, output_path, get_samples_history, dataset
    For MNIST:
      python SMGM_uncond.py generate --checkpoint_path "./checkpoints/unconditional/SM/mnist/" --num_samples 16 --reverse_steps 500 --schedule "linear" --interpolant "deterministic" --output_path "generated.png" --get_samples_history False --dataset "mnist"
    For CIFAR10:
      python SMGM_uncond.py generate --checkpoint_path "./checkpoints/unconditional/SM/cifar10/" --num_samples 16 --reverse_steps 500 --schedule "linear" --interpolant "deterministic" --output_path "generated.png" --get_samples_history False --dataset "cifat10"
  For fid:
    For MNIST:
      python SMGM_uncond.py fid --checkpoint_path "./checkpoints/unconditional/SM/mnist/" --num_samples 2500 --reverse_steps 500 --schedule "linear" --interpolant "deterministic" --dataset "mnist"
    For CIFAR10
      python SMGM_uncond.py fid --checkpoint_path "./checkpoints/unconditional/SM/cifar10/" --num_samples 2500 --reverse_steps 500 --schedule "linear" --interpolant "deterministic" --dataset "cifar10"
      
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

import utils.unet as unet

from torchmetrics.metric.fid import FrechetInceptionDistance

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

    def noise_loss(self, model, x_start, schedule, model_kwargs=None, **kwargs):
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
        predicted_noise = model(x_t, t_norm, **model_kwargs)
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

                beta_t = self.compute_beta_bar(t_norm_batch, schedule).view(-1, *([1] * (xt.dim() - 1)))

                f = -0.5 * beta_t * xt
                g = torch.sqrt(beta_t)

                # Get the score (using the nn)
                score = self.score_fn(model, xt, t_batch, schedule='linear')

                if interpolant == 'deterninistic':
                    xt = xt + (f - (g**2) * score / 2) * dt
                elif interpolant == 'stochastic':
                    z = torch.randn_like(xt)
                    xt = xt + (f - (g**2) * score) * dt + g * torch.sqrt(-dt) * z

                
                if get_sample_history:
                            all_images.append(xt.clone())
        return xt if not get_sample_history else torch.stack(all_images)
    

def create_unet(dataset):

    first_layer_embedding = False
    embedding_dim = 3 # MD4 needs a value for masks, so set of values is {0, 1, 2}
    output_dim = 1 # We only output a single probability value
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
            num_classes= None,
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


def train(num_epochs, checkpoint_interval, batch_size, learning_rate, checkpoint_dir, dataset, schedule):
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
        for batch_idx, (data, _) in (enumerate(dataloader)):
            data = data.to(device) 
            optimizer.zero_grad()

            # Compute the training loss.
            losses_dict = diffusion_process.noise_loss(model, x_start=data, schedule=schedule)
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

def generate(checkpoint_path, n_samples, reverse_steps, schedule, interpolant, output_path, get_samples_history, dataset):
    device = get_device()
    logging.info(f"Using device: {device}")

    # Create the class DiffusionProcess
    diffusion_process = DiffusionProcess(device=device)
    # the model will depend on the dataset used
    model = create_unet(dataset).to(device)

    # Load the checkpoint.
    with torch.inference_mode():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        samples = diffusion_process.sample(
            model=model,
            n_samples=n_samples,
            reverse_steps=reverse_steps,
            schedule=schedule,
            dataset=dataset,
            interpolant=interpolant,
            T=1,
            get_sample_history=False
        )

    # Save images
    save_images(samples, output_path, get_samples_history)
    logging.info(f"Saved generated images to {output_path}")

def save_images(samples, output_path, get_samples_history=False, dataset = None):
    """
    Save a grid of images to output_path. If the samples are 2D (as in a Gaussian mixture),
    we use matplotlib to create a scatter plot.
    If get_samples_history is True, also save the full history.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If samples are 2D points (either (N,2) or (steps, N,2)), use a scatter plot.
    float_samples = samples.clone().detach().float()
    
    if (float_samples.dim() == 2) or (float_samples.dim() == 3 and float_samples.shape[-1] == 2):
        # Assume shape (num_samples, 2) or (steps, num_samples, 2)
        if float_samples.dim() == 3:
            # Save final samples from the last time step.
            final_samples = float_samples[-1]
        else:
            final_samples = float_samples
        
        # clip finale samples to box [-lim, lim]\times [-lim, lim]
        box_lim = 1.2
        final_samples = torch.clamp(final_samples, -box_lim, box_lim)
        plt.figure(figsize=(6, 6))
        plt.scatter(final_samples[:, 0].cpu(), final_samples[:, 1].cpu(), s=10, alpha=0.6, label="Generated samples")
        if dataset is not None:
            # retrieve some samples from the original dataset
            real_samples = dataset[:final_samples.shape[0]]
            plt.scatter(real_samples[:, 0].cpu(), real_samples[:, 1].cpu(), s=10, alpha=0.6, label="Real samples")
        plt.title("Generated 2D Gaussian Mixture")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-box_lim, box_lim)
        plt.ylim(-box_lim, box_lim)
        plt.legend()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        if get_samples_history and float_samples.dim() == 3:
            history_dir = os.path.splitext(output_path)[0] + "_history"
            os.makedirs(history_dir, exist_ok=True)
            for i, step_samples in enumerate(float_samples):
                plt.figure(figsize=(6, 6))
                plt.scatter(step_samples[:, 0].cpu(), step_samples[:, 1].cpu(), s=10, alpha=0.6)
                plt.title(f"Step {i}")
                plt.xlabel("x")
                plt.ylabel("y")
                step_path = os.path.join(history_dir, f"step_{i:04d}.png")
                plt.savefig(step_path)
                plt.close()
    else:
        # Otherwise, assume images and use torchvision’s utility.
        if get_samples_history:
            final_samples = float_samples[-1]
            vutils.save_image(final_samples, output_path, nrow=int(math.sqrt(final_samples.size(0))), normalize=True)
            history_dir = os.path.splitext(output_path)[0] + "_history"
            os.makedirs(history_dir, exist_ok=True)
            for i, step_samples in enumerate(float_samples):
                step_path = os.path.join(history_dir, f"step_{i:04d}.png")
                vutils.save_image(step_samples, step_path, nrow=int(math.sqrt(step_samples.size(0))), normalize=True)
        else:
            vutils.save_image(float_samples, output_path, nrow=int(math.sqrt(float_samples.size(0))), normalize=True)



def compute_fid(checkpoint_path, num_samples, reverse_steps, schedule, interpolant, dataset):
    device = get_device()
    logging.info(f"Using device: {device}")

    # Create the class DiffusionProcess
    diffusion_process = DiffusionProcess(device=device)
    # the model will depend on the dataset used
    model = create_unet(dataset).to(device)

    # Load the checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    if dataset == 'mnist':
        # Transformation for the MNIST
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.clamp(0, 1))
        ])

        # Load the test dataset from MNIST
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    else:
        # Tranformation for the CIFAR10
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.clamp(0, 1))
        ])

        # Download the dataset
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    total_samples = num_samples
    batch_size = 64
    num_batches = total_samples // batch_size

    generated_images = []

    # Generate all batches
    for batch_idx in range(num_batches):
        logging.info(f"Generating batch {batch_idx + 1}/{num_batches}")
        samples = diffusion_process.sample(
            model=model,
            n_samples=batch_size,
            reverse_steps=reverse_steps,
            schedule=schedule,
            dataset=dataset,
            interpolant=interpolant,
            T=1,
            get_sample_history=False
        )
        final_step_images = torch.stack([torch.tensor(s).cpu() for s in samples])
        # Normalize images to [0, 1]
        final_step_images = (final_step_images - final_step_images.min()) / (final_step_images.max() - final_step_images.min())
        generated_images.append(final_step_images)

    # Stack all generated images
    generated_images = torch.cat(generated_images, dim=0)
    # print(f"Generated images shape: {generated_images.shape}")

    fid = FrechetInceptionDistance(feature=2048).to(device)


    for batch in test_loader:
        images, _ = batch
        if images.shape[1] == 1:  # if grayscale, repeat channels
            images = images.repeat(1, 3, 1, 1)
            # print(f"Fake:{images.min()}, {images.max()}")
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        fid.update(images.to(device), real=True)

    # Resize generated images
    generated_images = F.interpolate(generated_images, size=(299, 299), mode='bilinear', align_corners=False)

    # Make fake_loader
    fake_dataset = torch.utils.data.TensorDataset(generated_images)
    fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=64, shuffle=False)

    for batch in fake_loader:
        (images,) = batch
        if images.shape[1] == 1:  # if grayscale, repeat channels
            images = images.repeat(1, 3, 1, 1)
            # print(f"Fake:{images.min()}, {images.max()}")
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        fid.update(images.to(device), real=False)

    logging.info(f"FID Score: {fid_score.item()}")
    fid_score = fid.compute()


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("./fid/SMGM_deterministic.txt", mode='a'),       # saves to file # a for appending more info
            logging.StreamHandler()                        # prints to stdout
        ]
    )
    
    parser = argparse.ArgumentParser(description="Unconditional Score Matching Diffusion Training Script")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands: train, generate or fid")

    # Sub-parser for training.
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (N)")
    train_parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint interval (every C epochs)")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "cifar10"])
    train_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/unconditional/SM/", help="Directory to save checkpoints")
    train_parser.add_argument("--schedule", type=str, help="Noise schedule to use", choices=["linear", "cosine"])

    # Sub-parser for generation.
    gen_parser = subparsers.add_parser("generate", help="Generate images using a checkpoint")
    gen_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    gen_parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate ")
    gen_parser.add_argument("--reverse_steps", type=int, default=500, help="Number of reverse diffusion steps (T)")
    gen_parser.add_argument("--schedule", type=str, help="Schedule to use", choices=["linear", "cosine"])
    gen_parser.add_argument("--interpolant", type=str, help="Interpolant to use", choices=["deterministic", "stochastic"])
    gen_parser.add_argument("--output_path", type=str, default="generated.png", help="Path to save the generated image grid")
    gen_parser.add_argument("--get_samples_history", action="store_true", help="Save the full generation history")
    gen_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "cifar10"])

    # Sub-parser for fid
    gen_parser = subparsers.add_parser("fid", help="Compute the FID score")
    gen_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    gen_parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    gen_parser.add_argument("--reverse_steps", type=int, default=500, help="Number of reverse diffusion steps (T)")
    gen_parser.add_argument("--schedule", type=str, help="Schedule to use", choices=["linear", "cosine"])
    gen_parser.add_argument("--interpolant", type=str, help="Interpolant to use", choices=["deterministic", "stochastic"])
    gen_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "cifar10"])

    args = parser.parse_args()

    if args.command == "train":
        train(
            num_epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            dataset=args.dataset,
            schedule=args.schedule
            )
    elif args.command == "generate":
        generate(
            checkpoint_path=args.checkpoint_path,
            n_samples=args.n_samples,
            reverse_steps=args.reverse_steps,
            schedule=args.schedule,
            interpolant=args.interpolant,
            output_path=args.output_path,
            get_samples_history=args.get_samples_history,
            dataset=args.dataset
        )
    elif args.command == "fid":
        compute_fid(
            checkpoint_path=args.checkpoint_path,
            num_samples=args.num_samples,
            reverse_steps=args.reverse_steps,
            schedule=args.schedule,
            interpolant=args.interpolant,
            dataset=args.dataset
        )  
    
    parser.print_help()