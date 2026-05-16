"""
clean.py

This script cleans the DocRestoreAI project before uploading it to GitHub.

It moves large local artifacts such as:
- trained model checkpoints
- generated images
- restored PDFs
- evaluation generated images
- temporary output files

to a local backup folder called:

    _local_artifacts_backup/



The script does NOT delete files permanently.
It moves them to a backup folder.
"""

from pathlib import Path
import shutil
import os


# Get project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Move execution to project root
os.chdir(PROJECT_ROOT)

# Backup folder for local files that should not go to GitHub
BACKUP_DIR = PROJECT_ROOT / "_local_artifacts_backup"


def get_file_size_mb(file_path):
    """
    Returns file size in MB.

    Parameters:
        file_path: file path

    Returns:
        file size in MB
    """

    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def move_file_to_backup(file_path):
    """
    Moves a file to the local backup folder while preserving folder structure.

    Example:
        outputs/checkpoints/final_model.pth

    becomes:
        _local_artifacts_backup/outputs/checkpoints/final_model.pth

    Parameters:
        file_path: file that should be moved
    """

    # Convert to Path object
    file_path = Path(file_path)

    # Create relative path from project root
    relative_path = file_path.relative_to(PROJECT_ROOT)

    # Create destination path inside backup folder
    destination_path = BACKUP_DIR / relative_path

    # Create destination parent folder
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    # Move file to backup folder
    shutil.move(str(file_path), str(destination_path))

    print(f"[MOVED] {relative_path} -> {destination_path.relative_to(PROJECT_ROOT)}")


def clean_empty_folders(folder):
    """
    Removes empty folders inside a given directory.

    Parameters:
        folder: folder to clean
    """

    folder = Path(folder)

    if not folder.exists():
        return

    # Walk through folders from bottom to top
    for path in sorted(folder.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                # Folder is not empty, so we keep it
                pass


def create_gitkeep(folder):
    """
    Creates a .gitkeep file inside a folder.

    This allows GitHub to keep empty folders.

    Parameters:
        folder: folder where .gitkeep should be created
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    gitkeep_path = folder / ".gitkeep"
    gitkeep_path.touch(exist_ok=True)


def update_gitignore():
    """
    Adds safe ignore rules to .gitignore.

    These rules prevent large model files, datasets, and outputs
    from being uploaded to GitHub.
    """

    gitignore_path = PROJECT_ROOT / ".gitignore"

    block = """
# -------------------------------
# Local large files - do not upload
# -------------------------------

# Trained model checkpoints
*.pth
*.pt
*.h5
*.ckpt

# Local backup folder
_local_artifacts_backup/

# Dataset files
data/raw/*
data/clean/*
data/damaged/*
data/samples/*

# Keep dataset folder structure
!data/raw/.gitkeep
!data/clean/.gitkeep
!data/damaged/.gitkeep
!data/samples/.gitkeep

# Output files
outputs/restored_images/*
outputs/restored_pdfs/*
outputs/plots/*
outputs/metrics/*
outputs/checkpoints/*
outputs/generated_samples/*
outputs/evaluation/*

# Keep output folder structure
!outputs/restored_images/.gitkeep
!outputs/restored_pdfs/.gitkeep
!outputs/plots/.gitkeep
!outputs/metrics/.gitkeep
!outputs/checkpoints/.gitkeep
!outputs/generated_samples/.gitkeep
!outputs/evaluation/.gitkeep
"""

    # Read existing .gitignore content if it exists
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
    else:
        content = ""

    # Avoid adding the same block multiple times
    if "# Local large files - do not upload" not in content:
        with open(gitignore_path, "a", encoding="utf-8") as file:
            file.write("\n" + block.strip() + "\n")

        print("[OK] .gitignore updated.")
    else:
        print("[OK] .gitignore already contains large-file rules.")


def find_large_files(max_size_mb=90):
    """
    Finds files larger than max_size_mb.

    Parameters:
        max_size_mb: size threshold in MB

    Returns:
        list of large files
    """

    large_files = []

    # Skip virtual environment, git folder, and backup folder
    ignored_folders = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "_local_artifacts_backup"
    }

    for file_path in PROJECT_ROOT.rglob("*"):

        # Skip folders
        if file_path.is_dir():
            continue

        # Skip files inside ignored folders
        if any(part in ignored_folders for part in file_path.parts):
            continue

        # Check file size
        size_mb = get_file_size_mb(file_path)

        if size_mb >= max_size_mb:
            large_files.append((file_path, size_mb))

    return large_files


def move_known_artifacts():
    """
    Moves known heavy artifacts to the local backup folder.

    This includes model checkpoints and generated output files.
    """

    patterns = [
        "outputs/checkpoints/*.pth",
        "outputs/checkpoints/*.pt",
        "outputs/checkpoints/*.h5",
        "outputs/checkpoints/*.ckpt",
        "outputs/generated_samples/*",
        "outputs/restored_images/*",
        "outputs/restored_pdfs/*",
        "outputs/evaluation/generated_images/*",
        "outputs/evaluation/clean_images/*",
    ]

    moved_anything = False

    for pattern in patterns:
        for file_path in PROJECT_ROOT.glob(pattern):

            # Skip .gitkeep files
            if file_path.name == ".gitkeep":
                continue

            # Move only files
            if file_path.is_file():
                move_file_to_backup(file_path)
                moved_anything = True

    if not moved_anything:
        print("[OK] No known heavy artifacts found.")


def recreate_required_gitkeep_files():
    """
    Recreates .gitkeep files in important folders.
    """

    folders = [
        "data/raw",
        "data/clean",
        "data/damaged",
        "data/samples",
        "outputs/restored_images",
        "outputs/restored_pdfs",
        "outputs/plots",
        "outputs/metrics",
        "outputs/checkpoints",
        "outputs/generated_samples",
        "outputs/evaluation",
    ]

    for folder in folders:
        create_gitkeep(PROJECT_ROOT / folder)

    print("[OK] .gitkeep files created.")


def main():
    """
    Main cleanup function.
    """

    print("DocRestoreAI GitHub Cleanup")
    print("===========================")

    # Create backup folder
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Update .gitignore
    update_gitignore()

    print("\nMoving known heavy artifacts...")
    move_known_artifacts()

    print("\nChecking for files larger than 90 MB...")
    large_files = find_large_files(max_size_mb=90)

    if len(large_files) == 0:
        print("[OK] No large files found.")

    else:
        for file_path, size_mb in large_files:
            print(f"[LARGE] {file_path.relative_to(PROJECT_ROOT)} - {size_mb:.2f} MB")
            move_file_to_backup(file_path)

    print("\nCleaning empty folders...")
    clean_empty_folders(PROJECT_ROOT / "outputs")

    print("\nRecreating .gitkeep files...")
    recreate_required_gitkeep_files()

    print("\nCleanup completed.")
    print("Large local files were moved to:")
    print("_local_artifacts_backup/")
    print("\nNow run:")
    print("git status")


if __name__ == "__main__":
    main()