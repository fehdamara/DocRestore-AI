"""
inference.py

This file is used to restore a damaged document image or PDF using the trained Generator.

The model receives:
    - a damaged document image
    - or a damaged scanned PDF

The model produces:
    - a restored image
    - or a restored PDF

This file is important because it makes the project usable after training.
"""

# argparse allows us to pass input paths and options from the terminal
import argparse

# tempfile allows us to create temporary folders for PDF page processing
import tempfile

# Path is used to handle file paths in a clean way
from pathlib import Path

# PyTorch is used to load the model and run inference
import torch

# Import the Generator model
from src.models import UNetGenerator

# Import preprocessing and post-processing functions
from src.preprocessing import (
    read_image,
    save_image,
    resize_image,
    normalize_image,
    denormalize_image,
    image_to_tensor_format,
    tensor_to_image_format,
    classical_document_enhancement
)

# Import PDF utility functions
from src.pdf_utils import pdf_to_images, save_restored_pdf

# Import utility function to select GPU or CPU
from src.utils import get_device


def load_generator(checkpoint_path, device):
    """
    Loads the trained Generator from a checkpoint file.

    Parameters:
        checkpoint_path: path to the saved model checkpoint
        device: cpu or cuda

    Returns:
        generator: trained Generator model ready for inference
    """

    # Create the same Generator architecture used during training
    generator = UNetGenerator(
        in_channels=3,
        out_channels=3
    ).to(device)

    # Load checkpoint file
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Load only the Generator weights
    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    # Set the model to evaluation mode
    # This disables training-specific behavior such as dropout
    generator.eval()

    return generator


def restore_image_array(
    image_rgb,
    generator,
    device,
    image_size=256,
    apply_postprocessing=False
):
    """
    Restores one image using the trained Generator.

    Parameters:
        image_rgb: input damaged image in RGB format
        generator: trained Generator model
        device: cpu or cuda
        image_size: size expected by the model
        apply_postprocessing: whether to apply classical CV enhancement after generation

    Returns:
        restored_image: restored RGB image
    """

    # Save original image size
    # This allows us to resize the restored image back to its original dimensions
    original_height, original_width = image_rgb.shape[:2]

    # Resize image to the size used during training
    resized_image = resize_image(
        image_rgb,
        size=(image_size, image_size)
    )

    # Normalize image from [0, 255] to [-1, 1]
    normalized_image = normalize_image(resized_image)

    # Convert image from HWC format to CHW format
    tensor_image = image_to_tensor_format(normalized_image)

    # Convert NumPy array to PyTorch tensor
    input_tensor = torch.tensor(
        tensor_image,
        dtype=torch.float32
    )

    # Add batch dimension
    # Shape becomes [1, 3, image_size, image_size]
    input_tensor = input_tensor.unsqueeze(0)

    # Move tensor to CPU or GPU
    input_tensor = input_tensor.to(device)

    # Disable gradient calculation because we are not training here
    with torch.no_grad():

        # Generate restored image
        generated_tensor = generator(input_tensor)

    # Remove batch dimension
    generated_tensor = generated_tensor.squeeze(0)

    # Move generated tensor to CPU and convert it to NumPy
    generated_array = generated_tensor.cpu().numpy()

    # Convert from CHW back to HWC format
    generated_array = tensor_to_image_format(generated_array)

    # Denormalize from [-1, 1] back to [0, 255]
    restored_image = denormalize_image(generated_array)

    # Resize restored image back to the original input size
    restored_image = resize_image(
        restored_image,
        size=(original_width, original_height)
    )

    # Optional classical post-processing
    # This can improve text sharpness and contrast after generation
    if apply_postprocessing:
        restored_image = classical_document_enhancement(restored_image)

    return restored_image


def restore_image_file(
    input_image_path,
    output_image_path,
    generator,
    device,
    image_size=256,
    apply_postprocessing=False
):
    """
    Restores a single image file and saves the result.

    Parameters:
        input_image_path: path to damaged input image
        output_image_path: path where restored image will be saved
        generator: trained Generator model
        device: cpu or cuda
        image_size: model image size
        apply_postprocessing: whether to apply classical CV post-processing

    Returns:
        output_image_path: path to saved restored image
    """

    # Read damaged image from disk
    image_rgb = read_image(input_image_path)

    # Restore the image using the Generator
    restored_image = restore_image_array(
        image_rgb=image_rgb,
        generator=generator,
        device=device,
        image_size=image_size,
        apply_postprocessing=apply_postprocessing
    )

    # Save restored image to disk
    save_image(
        restored_image,
        output_image_path
    )

    return output_image_path


def restore_pdf_file(
    input_pdf_path,
    output_pdf_path,
    generator,
    device,
    image_size=256,
    apply_postprocessing=False
):
    """
    Restores a scanned PDF.

    The PDF workflow is:
        1. Convert PDF pages to images.
        2. Restore each page image with the Generator.
        3. Save restored page images.
        4. Combine restored images into one output PDF.

    Parameters:
        input_pdf_path: path to damaged PDF
        output_pdf_path: path where restored PDF will be saved
        generator: trained Generator model
        device: cpu or cuda
        image_size: model image size
        apply_postprocessing: whether to apply classical CV post-processing

    Returns:
        output_pdf_path: path to restored PDF
    """

    # Create a temporary directory
    # It will be automatically deleted after processing
    with tempfile.TemporaryDirectory() as temp_dir:

        # Create temporary paths for extracted and restored PDF pages
        temp_dir = Path(temp_dir)
        extracted_pages_dir = temp_dir / "extracted_pages"
        restored_pages_dir = temp_dir / "restored_pages"

        # Convert PDF pages into images
        page_image_paths = pdf_to_images(
            pdf_path=input_pdf_path,
            output_dir=extracted_pages_dir,
            dpi=200
        )

        # This list will store restored page images as NumPy arrays
        restored_images = []

        # Process each PDF page
        for page_path in page_image_paths:

            # Read page image
            page_image = read_image(page_path)

            # Restore page image
            restored_page = restore_image_array(
                image_rgb=page_image,
                generator=generator,
                device=device,
                image_size=image_size,
                apply_postprocessing=apply_postprocessing
            )

            # Add restored page to list
            restored_images.append(restored_page)

        # Save restored pages and combine them into a single PDF
        output_pdf_path = save_restored_pdf(
            restored_images=restored_images,
            output_dir=restored_pages_dir,
            output_pdf_path=output_pdf_path
        )

    return output_pdf_path


def is_pdf(path):
    """
    Checks if a file is a PDF.

    Parameters:
        path: file path

    Returns:
        True if the file extension is .pdf, otherwise False
    """

    return Path(path).suffix.lower() == ".pdf"


def is_image(path):
    """
    Checks if a file is a supported image.

    Parameters:
        path: file path

    Returns:
        True if file extension is an image extension
    """

    # Supported image extensions
    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    ]

    return Path(path).suffix.lower() in image_extensions


def create_default_output_path(input_path):
    """
    Creates a default output path if the user does not provide one.

    Example:
        input:  document.pdf
        output: outputs/restored_pdfs/document_restored.pdf

        input:  scan.png
        output: outputs/restored_images/scan_restored.png

    Parameters:
        input_path: input image or PDF path

    Returns:
        output_path: default output path
    """

    # Convert input path to Path object
    input_path = Path(input_path)

    # If input is PDF, save output in restored_pdfs folder
    if is_pdf(input_path):
        output_path = Path("outputs/restored_pdfs") / f"{input_path.stem}_restored.pdf"

    # Otherwise save output in restored_images folder
    else:
        output_path = Path("outputs/restored_images") / f"{input_path.stem}_restored.png"

    return output_path


def main(args):
    """
    Main inference function.

    This function:
        1. Selects CPU or GPU.
        2. Loads the trained Generator.
        3. Checks if input is image or PDF.
        4. Runs restoration.
        5. Saves the output.
    """

    # Select available device
    device = get_device()

    # Print selected device
    print(f"Using device: {device}")

    # Convert input path to Path object
    input_path = Path(args.input_path)

    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create default output path if none is provided
    if args.output_path is None:
        output_path = create_default_output_path(input_path)
    else:
        output_path = Path(args.output_path)

    # Create output folder if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load trained Generator model
    generator = load_generator(
        checkpoint_path=args.checkpoint,
        device=device
    )

    # Process PDF input
    if is_pdf(input_path):

        print("PDF input detected.")
        print("Restoring PDF pages...")

        restore_pdf_file(
            input_pdf_path=input_path,
            output_pdf_path=output_path,
            generator=generator,
            device=device,
            image_size=args.image_size,
            apply_postprocessing=args.postprocess
        )

    # Process image input
    elif is_image(input_path):

        print("Image input detected.")
        print("Restoring image...")

        restore_image_file(
            input_image_path=input_path,
            output_image_path=output_path,
            generator=generator,
            device=device,
            image_size=args.image_size,
            apply_postprocessing=args.postprocess
        )

    # Unsupported file type
    else:
        raise ValueError(
            "Unsupported input file type. Please use an image or PDF file."
        )

    print("Restoration completed successfully.")
    print(f"Output saved at: {output_path}")


def parse_args():
    """
    Defines command-line arguments for inference.

    This allows the user to run restoration from the terminal.
    """

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Restore damaged document images or PDFs using DocRestoreAI"
    )

    # Input file path
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to damaged input image or PDF"
    )

    # Model checkpoint path
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/final_model.pth",
        help="Path to trained model checkpoint"
    )

    # Optional output file path
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path where restored image or PDF will be saved"
    )

    # Image size used by the model
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size used by the trained model"
    )

    # Optional classical post-processing
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply classical CV post-processing after model output"
    )

    # Return parsed arguments
    return parser.parse_args()


# This block runs only when executing this file directly
if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_args()

    # Run inference
    main(args)