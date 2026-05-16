"""
utils.py

This file contains utility functions used across the project.

Main responsibilities:
1. Set random seeds for reproducibility.
2. Create folders automatically.
3. Save and load model checkpoints.
4. Convert tensors back to images.
5. Save sample training results.

These utilities keep the main training code cleaner and more modular.
"""

# Importiamo os per alcune operazioni sul sistema operativo
import os

# Importiamo random per controllare la casualità di Python
import random

# Importiamo NumPy per controllare la casualità di NumPy
import numpy as np

# Importiamo PyTorch per salvare modelli e gestire tensori
import torch

# Importiamo matplotlib per salvare immagini di confronto
import matplotlib.pyplot as plt

# Path permette di gestire percorsi in modo pulito
from pathlib import Path


def set_seed(seed=42):
    """
    Sets random seeds to make experiments more reproducible.

    Reproducibility is important because training neural networks includes
    random operations such as weight initialization and data shuffling.

    Parameters:
        seed: integer used as random seed
    """

    # Set seed for Python random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch CPU operations
    torch.manual_seed(seed)

    # Set seed for PyTorch GPU operations, if a GPU is available
    torch.cuda.manual_seed_all(seed)

    # This setting can make results more reproducible
    torch.backends.cudnn.deterministic = True

    # This disables some automatic optimizations to improve reproducibility
    torch.backends.cudnn.benchmark = False


def create_dir(path):
    """
    Creates a directory if it does not already exist.

    Parameters:
        path: folder path to create
    """

    # Convert path to Path object
    path = Path(path)

    # Create the folder and all missing parent folders
    path.mkdir(parents=True, exist_ok=True)


def get_device():
    """
    Selects the best available device.

    Returns:
        "cuda" if GPU is available, otherwise "cpu"
    """

    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, checkpoint_path):
    """
    Saves the training state to disk.

    This allows us to continue training later without starting from zero.

    Parameters:
        generator: trained Generator model
        discriminator: trained Discriminator model
        optimizer_g: optimizer for Generator
        optimizer_d: optimizer for Discriminator
        epoch: current training epoch
        checkpoint_path: path where checkpoint will be saved
    """

    # Convert checkpoint path to Path object
    checkpoint_path = Path(checkpoint_path)

    # Create parent directory if needed
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dictionary containing all important training information
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
    }

    # Save checkpoint to disk
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(generator, discriminator, optimizer_g, optimizer_d, checkpoint_path, device):
    """
    Loads a saved checkpoint.

    Parameters:
        generator: Generator model object
        discriminator: Discriminator model object
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        checkpoint_path: path of saved checkpoint
        device: cpu or cuda

    Returns:
        start_epoch: epoch number to continue from
    """

    # Load checkpoint from disk
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load Generator weights
    generator.load_state_dict(checkpoint["generator_state_dict"])

    # Load Discriminator weights
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    # Load optimizer states
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

    # Continue from next epoch
    start_epoch = checkpoint["epoch"] + 1

    return start_epoch


def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor image into a NumPy image.

    The model uses images in this format:
        [channels, height, width]
        values between -1 and 1

    For saving and visualization, we need:
        [height, width, channels]
        values between 0 and 1

    Parameters:
        tensor: image tensor

    Returns:
        image: NumPy image ready for matplotlib
    """

    # Move tensor from GPU to CPU and detach it from computation graph
    image = tensor.detach().cpu()

    # Convert from CHW to HWC format
    image = image.permute(1, 2, 0)

    # Convert from [-1, 1] to [0, 1]
    image = (image + 1.0) / 2.0

    # Clip values to avoid invalid display values
    image = torch.clamp(image, 0.0, 1.0)

    # Convert tensor to NumPy array
    image = image.numpy()

    return image


def save_sample_images(damaged, generated, clean, output_path, max_images=3):
    """
    Saves visual comparison images during training.

    The saved figure shows:
        damaged input | generated restoration | clean target

    This helps us check visually if the model is improving.

    Parameters:
        damaged: batch of damaged input images
        generated: batch of generated restored images
        clean: batch of clean target images
        output_path: path where the comparison image will be saved
        max_images: maximum number of samples to show
    """

    # Convert output path to Path object
    output_path = Path(output_path)

    # Create parent folder if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select number of images to display
    num_images = min(max_images, damaged.size(0))

    # Create a figure with rows = number of images and columns = 3
    fig, axes = plt.subplots(num_images, 3, figsize=(9, 3 * num_images))

    # If there is only one image, matplotlib returns a 1D axis array
    # This ensures axes always behaves like a 2D array
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    # Loop through selected images
    for i in range(num_images):

        # Convert tensors to displayable images
        damaged_img = tensor_to_image(damaged[i])
        generated_img = tensor_to_image(generated[i])
        clean_img = tensor_to_image(clean[i])

        # Show damaged image
        axes[i, 0].imshow(damaged_img)
        axes[i, 0].set_title("Damaged Input")
        axes[i, 0].axis("off")

        # Show generated restored image
        axes[i, 1].imshow(generated_img)
        axes[i, 1].set_title("Generated Output")
        axes[i, 1].axis("off")

        # Show clean target image
        axes[i, 2].imshow(clean_img)
        axes[i, 2].set_title("Clean Target")
        axes[i, 2].axis("off")

    # Improve layout spacing
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path)

    # Close figure to free memory
    plt.close(fig)


def save_training_log(log_path, text):
    """
    Appends one line of training information to a log file.

    Parameters:
        log_path: path to the log file
        text: text line to write
    """

    # Convert path to Path object
    log_path = Path(log_path)

    # Create parent folder if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Open file in append mode and write text
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(text + "\n")