"""
dataset.py

This file defines the PyTorch Dataset used for training the document restoration model.

The model needs paired images:

    damaged document image  ->  clean document image

There are two possible ways to provide pairs:

1. Use a folder of clean images and generate damaged images automatically.
2. Use two folders:
   - one folder with damaged images
   - one folder with clean target images

For this project, generating damaged images automatically is useful because
paired document restoration datasets are not always easy to find.
"""

# Path is used to work with file and folder paths
from pathlib import Path

# NumPy is used for numerical image processing
import numpy as np

# PyTorch is used to create Dataset and DataLoader
import torch
from torch.utils.data import Dataset, DataLoader

# Import preprocessing functions
from src.preprocessing import (
    read_image,
    resize_image,
    normalize_image,
    image_to_tensor_format
)

# Import degradation function to generate damaged documents
from src.degradation import damage_document


def collect_image_paths(folder):
    """
    Collects all image paths from a folder.

    Supported image formats:
        - jpg
        - jpeg
        - png
        - bmp
        - tif
        - tiff

    Parameters:
        folder: folder containing images

    Returns:
        image_paths: sorted list of image paths
    """

    # Convert folder to Path object
    folder = Path(folder)

    # Define supported image extensions
    supported_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff"
    ]

    # Create empty list for image paths
    image_paths = []

    # Search for each supported extension
    for extension in supported_extensions:
        image_paths.extend(folder.glob(extension))

    # Sort paths alphabetically for stable loading
    image_paths = sorted(image_paths)

    return image_paths


class DocumentRestorationDataset(Dataset):
    """
    PyTorch Dataset for document restoration.

    Each item returned by the dataset contains:
        - damaged image tensor
        - clean image tensor
        - image filename

    The damaged image is the model input.
    The clean image is the target output.
    """

    def __init__(
        self,
        clean_dir,
        damaged_dir=None,
        image_size=256,
        generate_damage=True
    ):
        """
        Initializes the dataset.

        Parameters:
            clean_dir: folder containing clean document images
            damaged_dir: optional folder containing damaged document images
            image_size: size used to resize images
            generate_damage: if True, damaged images are created automatically
        """

        # Store clean image directory
        self.clean_dir = Path(clean_dir)

        # Store optional damaged image directory
        self.damaged_dir = Path(damaged_dir) if damaged_dir is not None else None

        # Store target image size
        self.image_size = image_size

        # Store whether to generate damage automatically
        self.generate_damage = generate_damage

        # Collect all clean image paths
        self.clean_image_paths = collect_image_paths(self.clean_dir)

        # Raise an error if no clean images are found
        if len(self.clean_image_paths) == 0:
            raise ValueError(f"No clean images found in folder: {self.clean_dir}")

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            number of clean images
        """

        return len(self.clean_image_paths)

    def _find_matching_damaged_image(self, clean_image_path):
        """
        Finds a damaged image with the same filename as the clean image.

        This is used only if damaged_dir is provided and generate_damage is False.

        Parameters:
            clean_image_path: path to the clean image

        Returns:
            damaged_image_path: matching damaged image path or None
        """

        # If no damaged directory is provided, return None
        if self.damaged_dir is None:
            return None

        # Create possible damaged image path with same filename
        damaged_image_path = self.damaged_dir / clean_image_path.name

        # Return the path if it exists
        if damaged_image_path.exists():
            return damaged_image_path

        # Return None if no matching file exists
        return None

    def __getitem__(self, index):
        """
        Loads one sample from the dataset.

        Parameters:
            index: index of the sample

        Returns:
            sample: dictionary containing damaged image, clean image, and filename
        """

        # Get clean image path
        clean_image_path = self.clean_image_paths[index]

        # Read clean image from disk
        clean_image = read_image(clean_image_path)

        # Resize clean image to fixed size
        clean_image = resize_image(
            clean_image,
            size=(self.image_size, self.image_size)
        )

        # Decide whether to generate damaged image automatically
        if self.generate_damage:

            # Create a damaged version of the clean image using synthetic degradation
            damaged_image = damage_document(clean_image)

        else:
            # Try to find a real damaged image with the same filename
            damaged_image_path = self._find_matching_damaged_image(clean_image_path)

            # If matching damaged image exists, load it
            if damaged_image_path is not None:
                damaged_image = read_image(damaged_image_path)

                # Resize damaged image to fixed size
                damaged_image = resize_image(
                    damaged_image,
                    size=(self.image_size, self.image_size)
                )

            else:
                # If no matching damaged image exists, generate damage automatically
                damaged_image = damage_document(clean_image)

        # Normalize damaged image to range [-1, 1]
        damaged_image = normalize_image(damaged_image)

        # Normalize clean image to range [-1, 1]
        clean_image = normalize_image(clean_image)

        # Convert damaged image from HWC to CHW format
        damaged_image = image_to_tensor_format(damaged_image)

        # Convert clean image from HWC to CHW format
        clean_image = image_to_tensor_format(clean_image)

        # Convert NumPy arrays to PyTorch tensors
        damaged_tensor = torch.tensor(damaged_image, dtype=torch.float32)

        # Convert clean image to PyTorch tensor
        clean_tensor = torch.tensor(clean_image, dtype=torch.float32)

        # Create sample dictionary
        sample = {
            "damaged": damaged_tensor,
            "clean": clean_tensor,
            "filename": clean_image_path.name
        }

        return sample


def create_dataloader(
    clean_dir,
    damaged_dir=None,
    image_size=256,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    generate_damage=True
):
    """
    Creates a PyTorch DataLoader for training or validation.

    Parameters:
        clean_dir: folder with clean images
        damaged_dir: optional folder with damaged images
        image_size: image resize size
        batch_size: number of samples per batch
        shuffle: whether to shuffle the dataset
        num_workers: number of CPU workers
        generate_damage: whether to generate damage automatically

    Returns:
        dataloader: PyTorch DataLoader
    """

    # Create dataset object
    dataset = DocumentRestorationDataset(
        clean_dir=clean_dir,
        damaged_dir=damaged_dir,
        image_size=image_size,
        generate_damage=generate_damage
    )

    # Create DataLoader object
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader