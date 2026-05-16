"""
train.py

This file trains the Pix2Pix GAN for document restoration.

The model learns this mapping:

    damaged document image  ->  clean restored document image

Training has two neural networks:

1. Generator:
   - Takes a damaged document image.
   - Produces a restored document image.

2. Discriminator:
   - Receives the damaged image and either the real clean image or generated image.
   - Tries to decide if the restored image is real or fake.

Losses used:
1. Adversarial loss:
   - Makes generated images look realistic.

2. L1 reconstruction loss:
   - Makes generated images close to the clean target image.

The final Generator loss is:

    Generator Loss = GAN Loss + lambda_L1 * L1 Loss
"""

# argparse is used to pass settings from the command line
import argparse

# PyTorch main library
import torch

# Neural network modules and losses
import torch.nn as nn

# Optimizers for training
import torch.optim as optim

# tqdm shows a progress bar during training
from tqdm import tqdm

# Import DataLoader creation function
from src.dataset import create_dataloader

# Import Generator and Discriminator models
from src.models import UNetGenerator, PatchGANDiscriminator

# Import utility functions
from src.utils import (
    set_seed,
    get_device,
    create_dir,
    save_checkpoint,
    load_checkpoint,
    save_sample_images,
    save_training_log
)


def train_one_epoch(
    generator,
    discriminator,
    dataloader,
    optimizer_g,
    optimizer_d,
    adversarial_loss,
    l1_loss,
    device,
    lambda_l1,
    epoch,
    sample_output_dir
):
    """
    Trains the Generator and Discriminator for one epoch.

    Parameters:
        generator: U-Net Generator model
        discriminator: PatchGAN Discriminator model
        dataloader: PyTorch DataLoader
        optimizer_g: optimizer for Generator
        optimizer_d: optimizer for Discriminator
        adversarial_loss: GAN loss function
        l1_loss: reconstruction loss function
        device: cpu or cuda
        lambda_l1: weight for L1 loss
        epoch: current epoch number
        sample_output_dir: folder where sample images are saved

    Returns:
        avg_g_loss: average Generator loss
        avg_d_loss: average Discriminator loss
    """

    # Set models to training mode
    generator.train()
    discriminator.train()

    # Variables used to calculate average losses
    total_g_loss = 0.0
    total_d_loss = 0.0

    # tqdm creates a progress bar for the dataloader
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

    # Loop through all batches
    for batch_index, batch in enumerate(progress_bar):

        # Get damaged input images and move them to device
        damaged = batch["damaged"].to(device)

        # Get clean target images and move them to device
        clean = batch["clean"].to(device)

        # ----------------------------------------
        # 1. Train Discriminator
        # ----------------------------------------

        # Generate restored images using the Generator
        generated = generator(damaged)

        # Discriminator prediction for real image pairs
        pred_real = discriminator(damaged, clean)

        # Create labels for real images
        # torch.ones_like creates a tensor of ones with same shape as pred_real
        real_labels = torch.ones_like(pred_real, device=device)

        # Real loss: Discriminator should classify real pairs as real
        loss_d_real = adversarial_loss(pred_real, real_labels)

        # Discriminator prediction for fake image pairs
        # generated.detach() prevents gradients from updating Generator here
        pred_fake = discriminator(damaged, generated.detach())

        # Create labels for fake images
        fake_labels = torch.zeros_like(pred_fake, device=device)

        # Fake loss: Discriminator should classify generated pairs as fake
        loss_d_fake = adversarial_loss(pred_fake, fake_labels)

        # Final Discriminator loss is average of real and fake loss
        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        # Clear previous gradients for Discriminator
        optimizer_d.zero_grad()

        # Backpropagate Discriminator loss
        loss_d.backward()

        # Update Discriminator weights
        optimizer_d.step()

        # ----------------------------------------
        # 2. Train Generator
        # ----------------------------------------

        # Generate restored images again
        # This time we want gradients to update the Generator
        generated = generator(damaged)

        # Discriminator prediction for generated images
        pred_fake_for_g = discriminator(damaged, generated)

        # Generator wants Discriminator to classify generated images as real
        real_labels_for_g = torch.ones_like(pred_fake_for_g, device=device)

        # GAN loss for Generator
        loss_g_gan = adversarial_loss(pred_fake_for_g, real_labels_for_g)

        # L1 loss forces generated image to be close to clean target image
        loss_g_l1 = l1_loss(generated, clean)

        # Final Generator loss
        loss_g = loss_g_gan + (lambda_l1 * loss_g_l1)

        # Clear previous gradients for Generator
        optimizer_g.zero_grad()

        # Backpropagate Generator loss
        loss_g.backward()

        # Update Generator weights
        optimizer_g.step()

        # Add current losses to totals
        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()

        # Update progress bar text
        progress_bar.set_postfix({
            "G Loss": f"{loss_g.item():.4f}",
            "D Loss": f"{loss_d.item():.4f}"
        })

        # Save sample images from the first batch of each epoch
        if batch_index == 0:
            sample_path = f"{sample_output_dir}/epoch_{epoch:03d}.png"
            save_sample_images(
                damaged=damaged,
                generated=generated,
                clean=clean,
                output_path=sample_path,
                max_images=3
            )

    # Calculate average losses
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)

    return avg_g_loss, avg_d_loss


def main(args):
    """
    Main training function.

    This function:
    1. Sets up device and random seed.
    2. Creates folders.
    3. Loads dataset.
    4. Creates Generator and Discriminator.
    5. Defines losses and optimizers.
    6. Runs the training loop.
    7. Saves checkpoints and logs.
    """

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Select device: GPU if available, otherwise CPU
    device = get_device()

    # Print selected device
    print(f"Using device: {device}")

    # Create output folders
    create_dir(args.checkpoint_dir)
    create_dir(args.sample_output_dir)
    create_dir(args.log_dir)

    # Create DataLoader
    dataloader = create_dataloader(
        clean_dir=args.clean_dir,
        damaged_dir=args.damaged_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generate_damage=args.generate_damage
    )

    # Create Generator model and move it to device
    generator = UNetGenerator(
        in_channels=3,
        out_channels=3
    ).to(device)

    # Create Discriminator model and move it to device
    discriminator = PatchGANDiscriminator(
        in_channels=3
    ).to(device)

    # BCEWithLogitsLoss is used because Discriminator output has no sigmoid
    adversarial_loss = nn.BCEWithLogitsLoss()

    # L1Loss is used because it reduces blurriness compared to MSE
    l1_loss = nn.L1Loss()

    # Adam optimizer for Generator
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    # Adam optimizer for Discriminator
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    # Start training from epoch 1 by default
    start_epoch = 1

    # If resume checkpoint is provided, load it
    if args.resume_checkpoint is not None:
        print(f"Loading checkpoint: {args.resume_checkpoint}")
        start_epoch = load_checkpoint(
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            checkpoint_path=args.resume_checkpoint,
            device=device
        )

    # Path for training log
    log_path = f"{args.log_dir}/training_log.txt"

    # Write initial log line
    save_training_log(
        log_path,
        "epoch,generator_loss,discriminator_loss"
    )

    # Main training loop
    for epoch in range(start_epoch, args.epochs + 1):

        # Train one epoch
        avg_g_loss, avg_d_loss = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            adversarial_loss=adversarial_loss,
            l1_loss=l1_loss,
            device=device,
            lambda_l1=args.lambda_l1,
            epoch=epoch,
            sample_output_dir=args.sample_output_dir
        )

        # Print epoch summary
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Generator Loss: {avg_g_loss:.4f} "
            f"Discriminator Loss: {avg_d_loss:.4f}"
        )

        # Save loss values in log file
        save_training_log(
            log_path,
            f"{epoch},{avg_g_loss:.6f},{avg_d_loss:.6f}"
        )

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint_path = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch:03d}.pth"
            save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                epoch=epoch,
                checkpoint_path=checkpoint_path
            )

    # Save final checkpoint
    final_checkpoint_path = f"{args.checkpoint_dir}/final_model.pth"
    save_checkpoint(
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        epoch=args.epochs,
        checkpoint_path=final_checkpoint_path
    )

    print("Training completed successfully.")
    print(f"Final model saved at: {final_checkpoint_path}")


def parse_args():
    """
    Defines command-line arguments for training.

    This makes the training script flexible.
    For example, we can change batch size or number of epochs without editing the code.
    """

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Train Pix2Pix GAN for document restoration"
    )

    # Dataset folders
    parser.add_argument(
        "--clean_dir",
        type=str,
        default="data/clean",
        help="Folder containing clean document images"
    )

    parser.add_argument(
        "--damaged_dir",
        type=str,
        default=None,
        help="Optional folder containing damaged document images"
    )

    # Image and training settings
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size used for training"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of images per batch"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0002,
        help="Learning rate for Adam optimizers"
    )

    parser.add_argument(
        "--lambda_l1",
        type=float,
        default=100.0,
        help="Weight of L1 reconstruction loss"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers. Use 0 on Windows if errors occur."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # Whether to generate artificial damage automatically
    parser.add_argument(
        "--generate_damage",
        action="store_true",
        default=True,
        help="Generate damaged images automatically from clean images"
    )

    # Output folders
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/checkpoints",
        help="Folder where model checkpoints are saved"
    )

    parser.add_argument(
        "--sample_output_dir",
        type=str,
        default="outputs/generated_samples",
        help="Folder where sample output images are saved"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="outputs/metrics",
        help="Folder where training logs are saved"
    )

    # Resume training from checkpoint
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint if you want to resume training"
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )

    # Return parsed arguments
    return parser.parse_args()


# This block runs only when this file is executed directly as a module
if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_args()

    # Start training
    main(args)