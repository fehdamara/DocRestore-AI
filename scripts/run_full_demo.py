"""
run_full_demo.py

This script runs a complete demo of the DocRestoreAI project.

It performs:
1. Creation of sample clean documents.
2. Creation of a damaged demo image.
3. Creation of a demo PDF.
4. Quick training test.
5. Image restoration test.
6. PDF restoration test.
7. Evaluation test.

This is useful before submitting the project or showing it during the oral exam.
"""

from pathlib import Path
import subprocess
import sys
import os


# Get project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Move terminal execution to project root
os.chdir(PROJECT_ROOT)


def run_command(command, description):
    """
    Runs a terminal command and stops if the command fails.

    Parameters:
        command: list of command parts
        description: explanation of what the command does
    """

    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print("Running:", " ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n[ERROR] Failed step: {description}")
        sys.exit(result.returncode)

    print(f"[OK] Completed: {description}")


def check_file(path):
    """
    Checks if an expected file exists.

    Parameters:
        path: file path to check
    """

    path = Path(path)

    if path.exists():
        print(f"[OK] File created: {path}")
    else:
        print(f"[WARNING] Expected file not found: {path}")


def main():
    """
    Runs the full DocRestoreAI demo pipeline.
    """

    print("DocRestoreAI Full Demo Pipeline")
    print("===============================")

    # 1. Create clean sample documents
    run_command(
        [sys.executable, "scripts/create_sample_documents.py"],
        "Step 1 - Creating synthetic clean document images"
    )

    # 2. Create damaged demo input image
    run_command(
        [sys.executable, "scripts/create_demo_input.py"],
        "Step 2 - Creating damaged demo image"
    )

    # 3. Create demo PDF from damaged image
    run_command(
        [sys.executable, "scripts/create_demo_pdf.py"],
        "Step 3 - Creating damaged demo PDF"
    )

    # 4. Run sanity check
    run_command(
        [sys.executable, "scripts/sanity_check.py"],
        "Step 4 - Running sanity check"
    )

    # 5. Quick training test
    # We use only 2 epochs to verify that training works.
    # For real results, increase the number of epochs later.
    run_command(
        [
            sys.executable,
            "-m",
            "src.train",
            "--epochs",
            "2",
            "--batch_size",
            "1"
        ],
        "Step 5 - Running quick training test"
    )

    # 6. Image inference test
    run_command(
        [
            sys.executable,
            "-m",
            "src.inference",
            "--input_path",
            "data/samples/test_document.png",
            "--output_path",
            "outputs/restored_images/test_document_restored.png",
            "--postprocess"
        ],
        "Step 6 - Restoring damaged image"
    )

    # 7. PDF inference test
    run_command(
        [
            sys.executable,
            "-m",
            "src.inference",
            "--input_path",
            "data/samples/test_scan.pdf",
            "--output_path",
            "outputs/restored_pdfs/restored_scan.pdf",
            "--postprocess"
        ],
        "Step 7 - Restoring damaged PDF"
    )

    # 8. Evaluation test
    run_command(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--max_samples",
            "5"
        ],
        "Step 8 - Running evaluation"
    )

    print("\n" + "=" * 70)
    print("Final output check")
    print("=" * 70)

    # Check important outputs
    check_file("data/samples/test_document.png")
    check_file("data/samples/test_scan.pdf")
    check_file("outputs/checkpoints/final_model.pth")
    check_file("outputs/restored_images/test_document_restored.png")
    check_file("outputs/restored_pdfs/restored_scan.pdf")
    check_file("outputs/evaluation/metrics/evaluation_summary.txt")
    check_file("outputs/evaluation/metrics/evaluation_metrics.csv")
    check_file("outputs/evaluation/plots/psnr_plot.png")
    check_file("outputs/evaluation/plots/ssim_plot.png")

    print("\nFull demo completed.")
    print("The project pipeline is working.")
    print("You can now open Streamlit with:")
    print("python -m streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()