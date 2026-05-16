"""
pdf_utils.py

This file contains utility functions for working with PDF files.

Main responsibilities:
1. Convert PDF pages into images.
2. Convert restored images back into a PDF.
3. Save processed PDF outputs.
4. Support the final Streamlit app and inference pipeline.

This version uses PyMuPDF for both PDF-to-image and image-to-PDF conversion.
This avoids Pillow JPEG plugin errors when saving PDFs.
"""

# fitz is the import name used by PyMuPDF
import fitz

# Path is used for clean file path handling
from pathlib import Path

# Import save_image from our preprocessing file
from src.preprocessing import save_image


def pdf_to_images(pdf_path, output_dir, dpi=200):
    """
    Converts each page of a PDF into an image.

    Parameters:
        pdf_path: path to the input PDF
        output_dir: folder where extracted page images will be saved
        dpi: rendering quality; higher DPI means better quality but larger files

    Returns:
        image_paths: list of image paths created from the PDF
    """

    # Convert paths to Path objects
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # Create output folder if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the PDF file
    pdf_document = fitz.open(str(pdf_path))

    # This list will store paths of generated page images
    image_paths = []

    # Calculate zoom factor from DPI
    # PDF default resolution is 72 DPI
    zoom = dpi / 72

    # Create transformation matrix for rendering
    matrix = fitz.Matrix(zoom, zoom)

    # Loop through all pages in the PDF
    for page_index in range(len(pdf_document)):

        # Load one page from the PDF
        page = pdf_document.load_page(page_index)

        # Render the page as an image
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        # Create output image path
        image_path = output_dir / f"page_{page_index + 1:03d}.png"

        # Save the rendered page as PNG
        pixmap.save(str(image_path))

        # Add image path to the list
        image_paths.append(image_path)

    # Close the PDF document
    pdf_document.close()

    return image_paths


def images_to_pdf(image_paths, output_pdf_path):
    """
    Converts a list of image files into a single PDF file using PyMuPDF.

    This method is more stable than Pillow for PDF creation because it avoids
    JPEG plugin errors.

    Parameters:
        image_paths: list of image paths
        output_pdf_path: final output PDF path

    Returns:
        output_pdf_path: path of the created PDF
    """

    # Convert output path to Path object
    output_pdf_path = Path(output_pdf_path)

    # Create output folder if it does not exist
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort image paths to preserve page order
    image_paths = sorted([Path(path) for path in image_paths])

    # If no image paths are provided, raise an error
    if len(image_paths) == 0:
        raise ValueError("No images provided to create PDF.")

    # Create a new empty PDF document
    pdf_document = fitz.open()

    # Loop through all image paths
    for image_path in image_paths:

        # Open image as Pixmap
        pixmap = fitz.Pixmap(str(image_path))

        # If image has alpha channel, convert it to RGB
        if pixmap.alpha:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)

        # Create a PDF page with the same size as the image
        page = pdf_document.new_page(
            width=pixmap.width,
            height=pixmap.height
        )

        # Define the rectangle where the image will be inserted
        image_rect = fitz.Rect(
            0,
            0,
            pixmap.width,
            pixmap.height
        )

        # Insert the image into the PDF page
        page.insert_image(
            image_rect,
            filename=str(image_path)
        )

        # Release Pixmap memory
        pixmap = None

    # Save the final PDF
    pdf_document.save(str(output_pdf_path))

    # Close the PDF document
    pdf_document.close()

    return output_pdf_path


def save_restored_pdf(restored_images, output_dir, output_pdf_path):
    """
    Saves restored RGB images and combines them into a PDF.

    Parameters:
        restored_images: list of restored images as RGB NumPy arrays
        output_dir: temporary folder where restored page images are saved
        output_pdf_path: final PDF output path

    Returns:
        output_pdf_path: path to the final restored PDF
    """

    # Convert output directory to Path object
    output_dir = Path(output_dir)

    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # This list will store paths of saved restored page images
    restored_image_paths = []

    # Loop through restored images
    for index, image in enumerate(restored_images):

        # Create path for each restored page
        image_path = output_dir / f"restored_page_{index + 1:03d}.png"

        # Save the restored image as PNG
        save_image(image, image_path)

        # Add saved image path to list
        restored_image_paths.append(image_path)

    # Combine restored page images into one PDF
    output_pdf_path = images_to_pdf(
        image_paths=restored_image_paths,
        output_pdf_path=output_pdf_path
    )

    return output_pdf_path


def get_pdf_page_count(pdf_path):
    """
    Returns the number of pages in a PDF.

    Parameters:
        pdf_path: path to the PDF file

    Returns:
        page_count: number of pages
    """

    # Open PDF
    pdf_document = fitz.open(str(pdf_path))

    # Get number of pages
    page_count = len(pdf_document)

    # Close PDF
    pdf_document.close()

    return page_count