"""
sanity_check.py

This script performs a final sanity check before submitting the project.

It verifies:
1. Required folders exist.
2. Required files exist.
3. Important Python libraries are installed.
4. Generator model runs correctly.
5. Discriminator model runs correctly.
6. Dataset folder contains at least one image.

This script is useful before uploading the project to GitHub.
"""

# Path is used to check files and folders
from pathlib import Path

# sys is used to fix Python import paths
import sys

# os is used to change the current working directory to the project root
import os

# importlib is used to check if Python packages are installed
import importlib

# torch is used to test the deep learning models
import torch


# ------------------------------------------------------------
# Fix project import path
# ------------------------------------------------------------

# Get the absolute path of the project root folder.
# Example:
# C:\Users\Fehd\Desktop\DocRestore AI
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add the project root folder to sys.path.
# This allows imports like:
# from src.models import UNetGenerator
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change current working directory to the project root.
# This makes all relative paths work correctly, even if the script is executed
# from inside the scripts folder.
os.chdir(PROJECT_ROOT)


def check_path_exists(path, path_type="file"):
    """
    Checks if a file or folder exists.

    Parameters:
        path: path to check
        path_type: "file" or "folder"

    Returns:
        True if path exists, False otherwise
    """

    # Convert the path string to a Path object
    path = Path(path)

    # Check if the path should be a folder
    if path_type == "folder":
        if path.exists() and path.is_dir():
            print(f"[OK] Folder exists: {path}")
            return True
        else:
            print(f"[MISSING] Folder missing: {path}")
            return False

    # Check if the path should be a file
    if path_type == "file":
        if path.exists() and path.is_file():
            print(f"[OK] File exists: {path}")
            return True
        else:
            print(f"[MISSING] File missing: {path}")
            return False

    # If path_type is not valid, return False
    print(f"[ERROR] Invalid path type: {path_type}")
    return False


def check_required_folders():
    """
    Checks all required project folders.

    Returns:
        True if all folders exist, False otherwise
    """

    print("\nChecking required folders...")

    # List of required folders for the final project structure
    required_folders = [
        "app",
        "data",
        "data/raw",
        "data/clean",
        "data/damaged",
        "data/samples",
        "docs",
        "outputs",
        "outputs/restored_images",
        "outputs/restored_pdfs",
        "outputs/plots",
        "outputs/metrics",
        "outputs/checkpoints",
        "outputs/generated_samples",
        "outputs/evaluation",
        "src",
        "src/models",
        "scripts"
    ]

    # Check every required folder
    results = [
        check_path_exists(folder, path_type="folder")
        for folder in required_folders
    ]

    # Return True only if all folders exist
    return all(results)


def check_required_files():
    """
    Checks all required project files.

    Returns:
        True if all required files exist, False otherwise
    """

    print("\nChecking required files...")

    # List of required files for the project
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "LICENSE",
        "src/__init__.py",
        "src/preprocessing.py",
        "src/degradation.py",
        "src/dataset.py",
        "src/feature_extraction.py",
        "src/train.py",
        "src/evaluate.py",
        "src/inference.py",
        "src/pdf_utils.py",
        "src/utils.py",
        "src/models/__init__.py",
        "src/models/generator.py",
        "src/models/discriminator.py",
        "app/streamlit_app.py",
        "docs/technical_analysis.md",
        "docs/create_technical_pdf.py"
    ]

    # Recommended file.
    # It is not required for code execution, but it is required for final delivery.
    recommended_files = [
        "docs/technical_analysis.pdf"
    ]

    # Check required files
    results = [
        check_path_exists(file, path_type="file")
        for file in required_files
    ]

    print("\nChecking recommended files...")

    # Check recommended files
    for file in recommended_files:
        check_path_exists(file, path_type="file")

    # Return True only if all required files exist
    return all(results)


def check_libraries():
    """
    Checks if important Python libraries are installed.

    Returns:
        True if all required libraries are installed, False otherwise
    """

    print("\nChecking Python libraries...")

    # Dictionary format:
    # "package name in requirements.txt": "import name in Python"
    required_libraries = {
        "opencv-python": "cv2",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "scikit-image": "skimage",
        "torch": "torch",
        "torchvision": "torchvision",
        "Pillow": "PIL",
        "streamlit": "streamlit",
        "PyMuPDF": "fitz",
        "reportlab": "reportlab"
    }

    # This list stores True/False results
    results = []

    # Try importing every required library
    for package_name, import_name in required_libraries.items():
        try:
            importlib.import_module(import_name)
            print(f"[OK] Library installed: {package_name}")
            results.append(True)

        except ImportError:
            print(f"[MISSING] Library not installed: {package_name}")
            results.append(False)

    # Return True only if all libraries are installed
    return all(results)


def check_models():
    """
    Checks whether Generator and Discriminator can run a forward pass.

    Returns:
        True if both models work, False otherwise
    """

    print("\nChecking Generator and Discriminator...")

    try:
        # Import project models after sys.path has been fixed
        from src.models import UNetGenerator, PatchGANDiscriminator

        # Create fake input image tensors.
        # Shape format is:
        # batch_size, channels, height, width
        damaged = torch.randn(1, 3, 256, 256)
        clean = torch.randn(1, 3, 256, 256)

        # Create Generator model
        generator = UNetGenerator(
            in_channels=3,
            out_channels=3
        )

        # Run one forward pass through the Generator
        generated = generator(damaged)

        # Check that Generator output has the same shape as input
        if generated.shape == damaged.shape:
            print(f"[OK] Generator output shape: {generated.shape}")
        else:
            print(f"[ERROR] Generator wrong output shape: {generated.shape}")
            return False

        # Create Discriminator model
        discriminator = PatchGANDiscriminator(
            in_channels=3
        )

        # Run one forward pass through the Discriminator
        prediction = discriminator(damaged, clean)

        # PatchGAN output is a patch-level map, for example [1, 1, 30, 30]
        print(f"[OK] Discriminator output shape: {prediction.shape}")

        return True

    except Exception as error:
        print("[ERROR] Model check failed.")
        print(f"Reason: {error}")
        return False


def check_dataset_images():
    """
    Checks if data/clean contains at least one image.

    The training script needs clean document images.

    Returns:
        True if at least one image exists, False otherwise
    """

    print("\nChecking dataset images...")

    # Folder where clean document images should be placed
    clean_dir = Path("data/clean")

    # Supported image extensions
    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff"
    ]

    # Collect all image paths
    image_paths = []

    for extension in extensions:
        image_paths.extend(clean_dir.glob(extension))

    # Check if at least one image exists
    if len(image_paths) == 0:
        print("[WARNING] No images found in data/clean/")
        print("Add clean document images before training.")
        return False

    print(f"[OK] Found {len(image_paths)} clean image(s) in data/clean/")
    return True


def main():
    """
    Runs all sanity checks and prints a final summary.
    """

    print("DocRestoreAI Final Sanity Check")
    print("===============================")

    # Run all checks
    folders_ok = check_required_folders()
    files_ok = check_required_files()
    libraries_ok = check_libraries()
    models_ok = check_models()
    dataset_ok = check_dataset_images()

    # Print final summary
    print("\nFinal Summary")
    print("-------------")
    print(f"Folders:   {'OK' if folders_ok else 'CHECK REQUIRED'}")
    print(f"Files:     {'OK' if files_ok else 'CHECK REQUIRED'}")
    print(f"Libraries: {'OK' if libraries_ok else 'CHECK REQUIRED'}")
    print(f"Models:    {'OK' if models_ok else 'CHECK REQUIRED'}")
    print(f"Dataset:   {'OK' if dataset_ok else 'ADD IMAGES'}")

    # Final message
    if folders_ok and files_ok and libraries_ok and models_ok:
        print("\nProject structure and code are ready.")
    else:
        print("\nSome checks failed. Fix the missing items before submission.")


if __name__ == "__main__":
    main()