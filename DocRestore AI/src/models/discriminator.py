"""
discriminator.py

This file contains the PatchGAN Discriminator used in Pix2Pix.

The Discriminator receives two images:
1. The damaged input document.
2. The target or generated restored document.

It then decides whether the restored image looks real or fake.

PatchGAN does not classify the whole image with one single value.
Instead, it classifies many small patches of the image.

This is useful for document restoration because local details matter:
- text edges
- background cleanliness
- noise patterns
- document texture
"""

# Importiamo PyTorch
import torch

# Importiamo i moduli per costruire reti neurali
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """
    One block of the PatchGAN Discriminator.

    Each block usually contains:
    - Convolution
    - Batch Normalization
    - LeakyReLU activation
    """

    def __init__(self, in_channels, out_channels, stride=2, use_batchnorm=True):
        """
        Initializes one discriminator block.

        Parameters:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: convolution stride
            use_batchnorm: whether to use BatchNorm
        """

        # Chiamiamo il costruttore della classe padre nn.Module
        super().__init__()

        # Lista dei layer del blocco
        layers = []

        # Convolution layer
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False
            )
        )

        # BatchNorm is skipped in the first layer, as in Pix2Pix
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        # LeakyReLU is commonly used in GAN discriminators
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Convertiamo la lista dei layer in un blocco sequenziale
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the discriminator block.

        Parameters:
            x: input tensor

        Returns:
            output tensor
        """

        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix.

    Input:
        damaged image + restored/clean image

    Output:
        matrix of realism scores

    The input images are concatenated along the channel dimension.

    Example:
        damaged image shape: [batch, 3, 256, 256]
        clean image shape:   [batch, 3, 256, 256]

        concatenated input:  [batch, 6, 256, 256]
    """

    def __init__(self, in_channels=3):
        """
        Initializes the PatchGAN Discriminator.

        Parameters:
            in_channels: number of channels in one image.
                         Since we concatenate two images, the first layer gets in_channels * 2.
        """

        # Chiamiamo il costruttore della classe padre nn.Module
        super().__init__()

        # The discriminator sees both input and output images
        combined_channels = in_channels * 2

        # First block does not use BatchNorm
        self.block1 = DiscriminatorBlock(
            in_channels=combined_channels,
            out_channels=64,
            stride=2,
            use_batchnorm=False
        )

        # Second block
        self.block2 = DiscriminatorBlock(
            in_channels=64,
            out_channels=128,
            stride=2,
            use_batchnorm=True
        )

        # Third block
        self.block3 = DiscriminatorBlock(
            in_channels=128,
            out_channels=256,
            stride=2,
            use_batchnorm=True
        )

        # Fourth block uses stride 1 to keep more spatial information
        self.block4 = DiscriminatorBlock(
            in_channels=256,
            out_channels=512,
            stride=1,
            use_batchnorm=True
        )

        # Final convolution produces a patch-level output map
        self.final = nn.Conv2d(
            in_channels=512,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=1
        )

    def forward(self, damaged_image, restored_image):
        """
        Forward pass of the PatchGAN Discriminator.

        Parameters:
            damaged_image: original damaged document tensor
            restored_image: clean target or generated restored image tensor

        Returns:
            output: patch-level realism score map
        """

        # Concatenate damaged input and restored/generated image
        # dim=1 means we concatenate along the channel dimension
        x = torch.cat([damaged_image, restored_image], dim=1)

        # Pass through discriminator blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Final patch-level prediction
        output = self.final(x)

        return output


def test_discriminator():
    """
    Small test function to verify that the Discriminator works.
    """

    # Create random damaged image tensor
    damaged = torch.randn(1, 3, 256, 256)

    # Create random restored image tensor
    restored = torch.randn(1, 3, 256, 256)

    # Create discriminator
    discriminator = PatchGANDiscriminator(in_channels=3)

    # Get discriminator output
    output = discriminator(damaged, restored)

    # Print shapes
    print("Damaged input shape:", damaged.shape)
    print("Restored input shape:", restored.shape)
    print("Discriminator output shape:", output.shape)


# This block runs only when the file is executed directly
if __name__ == "__main__":
    test_discriminator()