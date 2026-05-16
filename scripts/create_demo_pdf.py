"""
create_demo_pdf.py

This script creates a simple demo PDF from the damaged test image.

Input:
    data/samples/test_document.png

Output:
    data/samples/test_scan.pdf
"""

from pathlib import Path
import sys
import os
import fitz

# Fix project root path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)


def main():
    """
    Creates a PDF file from test_document.png.
    """

    # Input image
    image_path = Path("data/samples/test_document.png")

    # Output PDF
    pdf_path = Path("data/samples/test_scan.pdf")

    # Check if the image exists
    if not image_path.exists():
        raise FileNotFoundError(
            "Missing data/samples/test_document.png. "
            "Run first: python scripts/create_demo_input.py"
        )

    # Open image with PyMuPDF to get its size
    pixmap = fitz.Pixmap(str(image_path))

    # Create a new PDF
    pdf_document = fitz.open()

    # Create a page with the same size as the image
    page = pdf_document.new_page(
        width=pixmap.width,
        height=pixmap.height
    )

    # Define page rectangle
    rect = fitz.Rect(0, 0, pixmap.width, pixmap.height)

    # Insert image into PDF page
    page.insert_image(rect, filename=str(image_path))

    # Save PDF
    pdf_document.save(str(pdf_path))

    # Close PDF
    pdf_document.close()

    print(f"Demo PDF created: {pdf_path}")


if __name__ == "__main__":
    main()