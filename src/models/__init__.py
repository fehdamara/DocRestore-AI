"""
This file marks the models folder as a Python package.

It also allows cleaner imports, for example:

from src.models import UNetGenerator, PatchGANDiscriminator
"""

# Import the U-Net Generator
from .generator import UNetGenerator

# Import the PatchGAN Discriminator
from .discriminator import PatchGANDiscriminator