"""
create_sample_documents.py

This script creates sample clean document images for testing DocRestoreAI.

The generated images are simple white document pages with black text.
They are useful for:
1. Testing the dataset loader.
2. Testing the training pipeline.
3. Testing the degradation system.
4. Running the sanity check without warnings.

These are not real confidential documents.
They are synthetic examples created only for this educational project.
"""

# Path is used to create and manage folders
from pathlib import Path

# PIL is used to create images and draw text
from PIL import Image, ImageDraw, ImageFont


def create_document_image(output_path, title, paragraphs):
    """
    Creates one synthetic clean document image.

    Parameters:
        output_path: where the image will be saved
        title: document title
        paragraphs: list of text paragraphs
    """

    # Create a white A4-like image
    width = 900
    height = 1200
    image = Image.new("RGB", (width, height), color="white")

    # Create drawing object
    draw = ImageDraw.Draw(image)

    # Load default font
    # This avoids problems if custom fonts are not installed
    title_font = ImageFont.load_default()
    text_font = ImageFont.load_default()

    # Starting position
    x = 80
    y = 80

    # Draw title
    draw.text((x, y), title, fill="black", font=title_font)

    # Move down after title
    y += 60

    # Draw paragraphs line by line
    for paragraph in paragraphs:

        # Split paragraph into words
        words = paragraph.split()

        # Build lines manually so they do not go outside the page
        line = ""

        for word in words:
            test_line = line + word + " "

            # Check approximate line length
            if len(test_line) > 90:
                draw.text((x, y), line, fill="black", font=text_font)
                y += 25
                line = word + " "
            else:
                line = test_line

        # Draw remaining line
        if line:
            draw.text((x, y), line, fill="black", font=text_font)
            y += 35

        # Add space between paragraphs
        y += 20

    # Create output folder if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image
    image.save(output_path)


def main():
    """
    Creates multiple sample document images inside data/clean.
    """

    # Output folder for clean documents
    output_dir = Path("data/clean")

    # Create folder if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example document contents
    documents = [
        {
            "title": "Document Restoration Test Page 1",
            "paragraphs": [
                "This is a synthetic clean document created for a Computer Vision project.",
                "The purpose of this image is to test document restoration, preprocessing, and model training.",
                "DocRestoreAI learns how to transform damaged document images into clean readable outputs."
            ]
        },
        {
            "title": "Sample Office Document",
            "paragraphs": [
                "This document simulates a simple office report with clear printed text.",
                "The model will create a damaged version automatically during training.",
                "The clean image is used as the target output for the Pix2Pix Generator."
            ]
        },
        {
            "title": "Archive Document Example",
            "paragraphs": [
                "Old scanned documents often contain blur, shadows, noise, and low contrast.",
                "This project applies both classical Computer Vision and deep learning methods.",
                "The final system can restore document images and scanned PDF files."
            ]
        },
        {
            "title": "Educational Project Document",
            "paragraphs": [
                "This project demonstrates a full Computer Vision pipeline.",
                "The pipeline includes preprocessing, feature extraction, generative modeling, post-processing, and evaluation.",
                "The results are documented in a technical analysis PDF."
            ]
        },
        {
            "title": "PDF Enhancement Example",
            "paragraphs": [
                "The application can process images and scanned PDF pages.",
                "Each PDF page is converted into an image, restored, and saved again as a PDF.",
                "This makes the system useful for real-world document enhancement tasks."
            ]
        }
    ]

    # Create each document image
    for index, document in enumerate(documents, start=1):
        output_path = output_dir / f"document_{index:03d}.png"

        create_document_image(
            output_path=output_path,
            title=document["title"],
            paragraphs=document["paragraphs"]
        )

        print(f"Created: {output_path}")

    print("Sample clean documents created successfully.")


if __name__ == "__main__":
    main()