"""
generator.py

This file contains the Generator model used in the Pix2Pix GAN.

The Generator is based on a U-Net architecture.

U-Net is useful for image-to-image translation because:
1. The encoder compresses the input image and learns high-level features.
2. The decoder reconstructs the output image.
3. Skip connections copy useful details from encoder to decoder.

In this project:

Input:
    damaged document image

Output:
    restored clean document image
"""

# Importiamo PyTorch
import torch

# Importiamo i moduli necessari per costruire reti neurali
import torch.nn as nn


class DownBlock(nn.Module):
    """
    DownBlock represents one encoder block of the U-Net.

    It reduces the spatial size of the image and increases the number of channels.

    Example:
        Input shape:  [batch, 3, 256, 256]
        Output shape: [batch, 64, 128, 128]

    This helps the model learn more abstract features.
    """

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        """
        Initializes the DownBlock.

        Parameters:
            in_channels: number of input channels
            out_channels: number of output channels
            use_batchnorm: whether to use BatchNorm after convolution
        """

        # Chiamiamo il costruttore della classe padre nn.Module
        super().__init__()

        # Lista dei layer che compongono il blocco
        layers = []

        # Convolution layer:
        # kernel_size=4 and stride=2 reduce image size by half
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )

        # BatchNorm stabilizes training, but we skip it in the first layer
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        # LeakyReLU is commonly used in GAN encoders
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Convertiamo la lista di layer in un blocco sequenziale
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the DownBlock.

        Parameters:
            x: input tensor

        Returns:
            output tensor after convolution, normalization, and activation
        """

        return self.block(x)


class UpBlock(nn.Module):
    """
    UpBlock represents one decoder block of the U-Net.

    It increases the spatial size of the feature map and reduces the number of channels.

    Skip connections are applied outside this block in the Generator class.
    """

    def __init__(self, in_channels, out_channels, use_dropout=False):
        """
        Initializes the UpBlock.

        Parameters:
            in_channels: number of input channels
            out_channels: number of output channels
            use_dropout: whether to use dropout for regularization
        """

        # Chiamiamo il costruttore della classe padre nn.Module
        super().__init__()

        # Lista dei layer che compongono il blocco
        layers = []

        # ConvTranspose2d performs upsampling
        # stride=2 doubles the image size
        layers.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )

        # BatchNorm helps stabilize GAN training
        layers.append(nn.BatchNorm2d(out_channels))

        # ReLU is commonly used in the decoder
        layers.append(nn.ReLU(inplace=True))

        # Dropout helps prevent overfitting in deeper decoder layers
        if use_dropout:
            layers.append(nn.Dropout(0.5))

        # Convertiamo la lista in un blocco sequenziale
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the UpBlock.

        Parameters:
            x: input tensor

        Returns:
            output tensor after upsampling
        """

        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net Generator for document restoration.

    The model receives a damaged document image and generates a clean restored image.

    Architecture:
        Encoder:
            down1, down2, down3, down4, down5, down6, down7, bottleneck

        Decoder:
            up1, up2, up3, up4, up5, up6, up7, final

    Skip connections:
        Encoder features are concatenated with decoder features.
        This helps preserve document details such as text edges and layout.
    """

    def __init__(self, in_channels=3, out_channels=3):
        """
        Initializes the U-Net Generator.

        Parameters:
            in_channels: number of channels in the input image
            out_channels: number of channels in the output image
        """

        # Chiamiamo il costruttore della classe padre nn.Module
        super().__init__()

        # -------------------------
        # Encoder / Downsampling
        # -------------------------

        # First layer does not use BatchNorm in Pix2Pix
        self.down1 = DownBlock(in_channels, 64, use_batchnorm=False)

        # The following layers progressively reduce image size
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)

        # Bottleneck layer
        # No BatchNorm here to avoid problems with very small spatial dimensions
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # -------------------------
        # Decoder / Upsampling
        # -------------------------

        # The first decoder layers use dropout as in Pix2Pix
        self.up1 = UpBlock(512, 512, use_dropout=True)

        # Because of skip connections, input channels become doubled
        self.up2 = UpBlock(1024, 512, use_dropout=True)
        self.up3 = UpBlock(1024, 512, use_dropout=True)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)

        # Final layer converts feature map back to output image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Tanh outputs values between -1 and 1
            # This matches our normalization function
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass of the U-Net Generator.

        Parameters:
            x: damaged document image tensor

        Returns:
            restored image tensor
        """

        # -------------------------
        # Encoder forward pass
        # -------------------------

        # Each down block stores features for skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # Bottleneck represents the most compressed representation
        bottleneck = self.bottleneck(d7)

        # -------------------------
        # Decoder forward pass
        # -------------------------

        # up1 output is concatenated with d7
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d7], dim=1)

        # up2 output is concatenated with d6
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)

        # up3 output is concatenated with d5
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)

        # up4 output is concatenated with d4
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)

        # up5 output is concatenated with d3
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)

        # up6 output is concatenated with d2
        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)

        # up7 output is concatenated with d1
        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        # Final output image
        output = self.final(u7)

        return output


def test_generator():
    """
    Small test function to verify that the Generator works.

    This is useful for debugging before starting training.
    """

    # Create a random input tensor that simulates a batch of damaged images
    sample_input = torch.randn(1, 3, 256, 256)

    # Create the generator
    generator = UNetGenerator(in_channels=3, out_channels=3)

    # Generate output
    sample_output = generator(sample_input)

    # Print input and output shapes
    print("Generator input shape:", sample_input.shape)
    print("Generator output shape:", sample_output.shape)


# This block runs only when the file is executed directly
if __name__ == "__main__":
    test_generator()