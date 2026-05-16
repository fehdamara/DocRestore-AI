"""
create_demo_input.py

This script creates a damaged demo document from one clean document image.

It is useful for testing:
1. The degradation pipeline.
2. The inference script.
3. The Streamlit app.
4. The final project demonstration.

Input:
    data/clean/document_001.png

Output:
    data/samples/test_document.png
"""

from pathlib import Path
import sys
import os

# Fix project import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

# Import project functions
from src.preprocessing import read_image, save_image
from src.degradation import damage_document


def main():
    """
    Creates one damaged test image from a clean document.
    """

    # Input clean document
    input_path = Path("data/clean/document_001.png")

    # Output damaged document
    output_path = Path("data/samples/test_document.png")

    # Check if clean image exists
    if not input_path.exists():
        raise FileNotFoundError(
            "Clean sample not found. Run this first: python scripts/create_sample_documents.py"
        )

    # Read clean image
    clean_image = read_image(input_path)

    # Create damaged version
    damaged_image = damage_document(clean_image)

    # Save damaged image
    save_image(damaged_image, output_path)

    print(f"Demo damaged document created: {output_path}")


if __name__ == "__main__":
    main()