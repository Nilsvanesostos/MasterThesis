# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Generative Unconditional Bridge Matching Diffusion Model on simple datasets

This script demostrates:
  - Creating the diffusion process, U-Net model, and optimizer.
  - Loading MNIST or CIFAR10 datasets.
  - Training the model  for N epochs saving the at the end the model.

Usage:
  For MNIST:
    python BMGM_uncond.py --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 2e-3 --dataset mnist
  For CIFAR10:
    python BMGM_uncond.py --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 1e-4 --dataset cifar10
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

from torchmetrics.image.fid import FrechetInceptionDistance

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

    def loss_fn(self, model, x_start, model_kwargs=None, **kwargs):
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
        pred = model(x_t, t_norm, **model_kwargs)
        loss = F.mse_loss(pred, x_1)

        return {'loss': loss}

    def sample(self, model, n_samples, reverse_steps, dataset, get_sample_history=False,  **kwargs):
        """
        Smpling for the Flow Diffusion Model
        """
        if dataset == 'mnist':
            channels = 1
        elif dataset == 'cifar10':
            channels = 3

        shape = (n_samples, channels, 32, 32)
        sigma = 0.5
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
                t_comp = match_last_dims(t_batch, xt.shape)
                t_norm_batch = t_batch / self.T

                noise = torch.randn_like(xt)

                # Predict the clean x_t
                v = (model(xt, t_norm_batch) - xt) / (1 - t_comp + 1e-5)

                xt = xt + (self.T / reverse_steps) * v + np.sqrt(self.T / reverse_steps) * sigma * noise

                
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
        for batch_idx, (data, _) in (enumerate(dataloader)):
            data = data.to(device) 
            optimizer.zero_grad()

            # Compute the training loss.
            losses_dict = diffusion_process.loss_fn(model, x_start=data)
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

def generate(checkpoint_path, n_samples, reverse_steps, output_path, get_samples_history, dataset):
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
            dataset=dataset,
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



def compute_fid(checkpoint_path, num_samples, reverse_steps, dataset):
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
            dataset=dataset,
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
            logging.FileHandler("training_log.txt", mode='a'),       # saves to file # a for appending more info
            logging.StreamHandler()                        # prints to stdout
        ]
    )

    parser = argparse.ArgumentParser(description="Unconditional Bridge Matching Diffusion Training Script")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands: train, generate or fid")

    # Sub-parser for training.
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (N)")
    train_parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint interval (every C epochs)")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "cifar10"])
    train_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/unconditional/BM/", help="Directory to save checkpoints")

    # Sub-parser for generation.
    gen_parser = subparsers.add_parser("generate", help="Generate images using a checkpoint")
    gen_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    gen_parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate ")
    gen_parser.add_argument("--reverse_steps", type=int, default=500, help="Number of reverse diffusion steps (T)")
    gen_parser.add_argument("--output_path", type=str, default="generated.png", help="Path to save the generated image grid")
    gen_parser.add_argument("--get_samples_history", action="store_true", help="Save the full generation history")
    gen_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "cifar10"])

    # Sub-parser for fid
    gen_parser = subparsers.add_parser("fid", help="Compute the FID score")
    gen_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    gen_parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    gen_parser.add_argument("--reverse_steps", type=int, default=500, help="Number of reverse diffusion steps (T)") 
    gen_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "cifar10"])


    args = parser.parse_args()

    if args.command == "train":
        train(
            num_epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            dataset=args.dataset
            )
    elif args.command == "generate":
        generate(
            checkpoint_path=args.checkpoint_path,
            n_samples=args.n_samples,
            reverse_steps=args.reverse_steps,
            output_path=args.output_path,
            get_samples_history=args.get_samples_history,
            dataset=args.dataset
        )
    elif args.command == "fid":
        compute_fid(
            checkpoint_path=args.checkpoint_path,
            num_samples=args.num_samples,
            reverse_steps=args.reverse_steps,
            dataset=args.dataset
        )  
    
    parser.print_help()