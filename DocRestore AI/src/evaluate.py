"""
evaluate.py

This file evaluates the trained DocRestoreAI model.

The project is a generative image-to-image translation system.
Therefore, we evaluate it using generation and restoration metrics.

Metrics included:
1. PSNR
2. SSIM
3. Optional FID
4. Optional OCR readability comparison
5. Qualitative visual comparison

The evaluation results are saved into:
- CSV file
- TXT summary report
- plots
- qualitative comparison images
"""

# argparse is used for command-line arguments
import argparse

# Path is used for clean path handling
from pathlib import Path

# NumPy is used for numerical calculations
import numpy as np

# Pandas is used to save metrics in table format
import pandas as pd

# PyTorch is used to run the model
import torch

# Matplotlib is used to save evaluation plots
import matplotlib.pyplot as plt

# SequenceMatcher is used to compare OCR text similarity
from difflib import SequenceMatcher

# PSNR and SSIM are image quality metrics
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# Import dataset loader
from src.dataset import create_dataloader

# Import Generator model
from src.models import UNetGenerator

# Import preprocessing functions
from src.preprocessing import (
    denormalize_image,
    tensor_to_image_format,
    save_image
)

# Import feature comparison functions
from src.feature_extraction import compare_document_features

# Import utility functions
from src.utils import get_device, create_dir


def load_generator(checkpoint_path, device):
    """
    Loads the trained Generator model from checkpoint.

    Parameters:
        checkpoint_path: path to model checkpoint
        device: cpu or cuda

    Returns:
        generator: trained Generator model
    """

    # Create Generator architecture
    generator = UNetGenerator(
        in_channels=3,
        out_channels=3
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Load Generator weights
    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    # Set model to evaluation mode
    generator.eval()

    return generator


def tensor_to_uint8_image(tensor):
    """
    Converts a normalized PyTorch tensor into a uint8 RGB image.

    Input tensor format:
        channels x height x width

    Input value range:
        -1 to 1

    Output image format:
        height x width x channels

    Output value range:
        0 to 255

    Parameters:
        tensor: PyTorch image tensor

    Returns:
        image_uint8: RGB image as NumPy uint8 array
    """

    # Move tensor to CPU and detach from computation graph
    array = tensor.detach().cpu().numpy()

    # Convert from CHW to HWC
    array = tensor_to_image_format(array)

    # Convert from [-1, 1] to [0, 255]
    image_uint8 = denormalize_image(array)

    return image_uint8


def calculate_ssim(clean_image, generated_image):
    """
    Calculates SSIM between clean target and generated image.

    SSIM means Structural Similarity Index.
    Higher SSIM means the generated image is structurally closer to the target.

    Parameters:
        clean_image: ground truth image
        generated_image: generated restored image

    Returns:
        ssim_score: SSIM value
    """

    # Newer versions of scikit-image use channel_axis
    try:
        ssim_score = structural_similarity(
            clean_image,
            generated_image,
            channel_axis=2,
            data_range=255
        )

    # Older versions use multichannel
    except TypeError:
        ssim_score = structural_similarity(
            clean_image,
            generated_image,
            multichannel=True,
            data_range=255
        )

    return float(ssim_score)


def calculate_psnr(clean_image, generated_image):
    """
    Calculates PSNR between clean target and generated image.

    PSNR means Peak Signal-to-Noise Ratio.
    Higher PSNR usually means better reconstruction quality.

    Parameters:
        clean_image: ground truth image
        generated_image: generated restored image

    Returns:
        psnr_score: PSNR value
    """

    psnr_score = peak_signal_noise_ratio(
        clean_image,
        generated_image,
        data_range=255
    )

    return float(psnr_score)


def safe_ocr_text(image_rgb):
    """
    Runs OCR on an image using pytesseract.

    This function is called "safe" because it does not crash the evaluation
    if Tesseract is not installed on the computer.

    Parameters:
        image_rgb: RGB image

    Returns:
        extracted_text: OCR text or empty string if OCR fails
    """

    try:
        # Import pytesseract only when needed
        import pytesseract

        # Run OCR
        extracted_text = pytesseract.image_to_string(image_rgb)

        return extracted_text.strip()

    except Exception:
        # If OCR fails, return empty text
        return ""


def text_similarity(text_a, text_b):
    """
    Calculates similarity between two OCR texts.

    The score is between 0 and 1:
        0 = completely different
        1 = identical

    Parameters:
        text_a: first text
        text_b: second text

    Returns:
        similarity_score: similarity ratio
    """

    similarity_score = SequenceMatcher(
        None,
        text_a,
        text_b
    ).ratio()

    return float(similarity_score)


def calculate_ocr_metrics(damaged_image, generated_image, clean_image):
    """
    Calculates OCR readability improvement.

    Since we may not have original text labels, we use the OCR text from
    the clean image as a reference.

    Then we compare:
        OCR(damaged image)  vs OCR(clean image)
        OCR(generated image) vs OCR(clean image)

    If generated similarity is higher, restoration improved readability.

    Parameters:
        damaged_image: damaged input image
        generated_image: restored generated image
        clean_image: clean target image

    Returns:
        ocr_metrics: dictionary with OCR scores
    """

    # Extract OCR text from damaged image
    damaged_text = safe_ocr_text(damaged_image)

    # Extract OCR text from generated image
    generated_text = safe_ocr_text(generated_image)

    # Extract OCR text from clean target image
    clean_text = safe_ocr_text(clean_image)

    # If OCR cannot extract clean text, return empty metrics
    if len(clean_text) == 0:
        return {
            "ocr_before_similarity": None,
            "ocr_after_similarity": None,
            "ocr_improvement": None,
            "ocr_before_length": len(damaged_text),
            "ocr_after_length": len(generated_text),
            "ocr_clean_length": len(clean_text)
        }

    # Compare damaged OCR with clean OCR
    before_similarity = text_similarity(damaged_text, clean_text)

    # Compare generated OCR with clean OCR
    after_similarity = text_similarity(generated_text, clean_text)

    # Calculate improvement
    improvement = after_similarity - before_similarity

    return {
        "ocr_before_similarity": before_similarity,
        "ocr_after_similarity": after_similarity,
        "ocr_improvement": improvement,
        "ocr_before_length": len(damaged_text),
        "ocr_after_length": len(generated_text),
        "ocr_clean_length": len(clean_text)
    }


def save_qualitative_comparison(
    damaged_image,
    generated_image,
    clean_image,
    output_path
):
    """
    Saves a side-by-side comparison image.

    The comparison shows:
        damaged input | generated output | clean target

    This is important for qualitative evaluation.

    Parameters:
        damaged_image: input damaged image
        generated_image: model output
        clean_image: ground truth clean image
        output_path: where the comparison image will be saved
    """

    # Convert output path to Path object
    output_path = Path(output_path)

    # Create folder if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create figure with 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Show damaged input
    axes[0].imshow(damaged_image)
    axes[0].set_title("Damaged Input")
    axes[0].axis("off")

    # Show generated output
    axes[1].imshow(generated_image)
    axes[1].set_title("Generated Restoration")
    axes[1].axis("off")

    # Show clean target
    axes[2].imshow(clean_image)
    axes[2].set_title("Clean Target")
    axes[2].axis("off")

    # Improve layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path)

    # Close figure
    plt.close(fig)


def plot_metric_values(metrics_df, output_dir):
    """
    Saves plots for PSNR and SSIM values.

    Parameters:
        metrics_df: DataFrame containing evaluation metrics
        output_dir: folder where plots will be saved
    """

    # Convert output folder to Path object
    output_dir = Path(output_dir)

    # Create output folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot PSNR values
    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df["sample_index"], metrics_df["psnr"])
    plt.xlabel("Sample Index")
    plt.ylabel("PSNR")
    plt.title("PSNR per Test Sample")
    plt.tight_layout()
    plt.savefig(output_dir / "psnr_plot.png")
    plt.close()

    # Plot SSIM values
    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df["sample_index"], metrics_df["ssim"])
    plt.xlabel("Sample Index")
    plt.ylabel("SSIM")
    plt.title("SSIM per Test Sample")
    plt.tight_layout()
    plt.savefig(output_dir / "ssim_plot.png")
    plt.close()


def compute_optional_fid(real_dir, generated_dir):
    """
    Computes FID score using clean-fid if available.

    FID means Fréchet Inception Distance.
    Lower FID means generated images are closer to real images.

    This function is optional because clean-fid may need extra setup
    or model weights download.

    Parameters:
        real_dir: folder containing real clean images
        generated_dir: folder containing generated images

    Returns:
        fid_score: FID value or None if calculation fails
    """

    try:
        # Import clean-fid only when needed
        from cleanfid import fid

        # Compute FID
        fid_score = fid.compute_fid(
            str(real_dir),
            str(generated_dir)
        )

        return float(fid_score)

    except Exception as error:
        print("FID could not be computed.")
        print(f"Reason: {error}")

        return None


def write_summary_report(metrics_df, output_path, fid_score=None):
    """
    Writes a text summary report with average evaluation results.

    Parameters:
        metrics_df: DataFrame containing metric values
        output_path: path where TXT report will be saved
        fid_score: optional FID score
    """

    # Convert output path to Path object
    output_path = Path(output_path)

    # Create parent folder
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate average metrics
    avg_psnr = metrics_df["psnr"].mean()
    avg_ssim = metrics_df["ssim"].mean()

    # OCR columns may contain None values, so we check before calculating mean
    if "ocr_improvement" in metrics_df.columns:
        avg_ocr_improvement = metrics_df["ocr_improvement"].dropna().mean()
    else:
        avg_ocr_improvement = None

    # Write summary report
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("DocRestoreAI Evaluation Summary\n")
        file.write("================================\n\n")

        file.write(f"Number of evaluated samples: {len(metrics_df)}\n")
        file.write(f"Average PSNR: {avg_psnr:.4f}\n")
        file.write(f"Average SSIM: {avg_ssim:.4f}\n")

        if fid_score is not None:
            file.write(f"FID Score: {fid_score:.4f}\n")
        else:
            file.write("FID Score: Not computed\n")

        if avg_ocr_improvement is not None and not np.isnan(avg_ocr_improvement):
            file.write(f"Average OCR Improvement: {avg_ocr_improvement:.4f}\n")
        else:
            file.write("Average OCR Improvement: Not available\n")

        file.write("\nInterpretation:\n")
        file.write("- Higher PSNR means better pixel-level reconstruction.\n")
        file.write("- Higher SSIM means better structural similarity.\n")
        file.write("- Lower FID means generated images are closer to real clean documents.\n")
        file.write("- Positive OCR improvement means restored images are more readable.\n")


def evaluate_model(args):
    """
    Main evaluation function.

    This function:
    1. Loads the trained Generator.
    2. Loads evaluation images.
    3. Generates restored outputs.
    4. Calculates PSNR and SSIM.
    5. Optionally calculates OCR improvement.
    6. Optionally calculates FID.
    7. Saves tables, plots, and qualitative examples.
    """

    # Select CPU or GPU
    device = get_device()

    # Print selected device
    print(f"Using device: {device}")

    # Create output folders
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    qualitative_dir = output_dir / "qualitative"
    generated_dir = output_dir / "generated_images"
    clean_dir = output_dir / "clean_images"

    create_dir(metrics_dir)
    create_dir(plots_dir)
    create_dir(qualitative_dir)
    create_dir(generated_dir)
    create_dir(clean_dir)

    # Load trained Generator
    generator = load_generator(
        checkpoint_path=args.checkpoint,
        device=device
    )

    # Create evaluation dataloader
    dataloader = create_dataloader(
        clean_dir=args.clean_dir,
        damaged_dir=args.damaged_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generate_damage=args.generate_damage
    )

    # List that will store metrics for each sample
    metric_rows = []

    # Sample counter
    sample_index = 0

    # Disable gradients because we are evaluating, not training
    with torch.no_grad():

        # Loop through batches
        for batch in dataloader:

            # Move input and target to device
            damaged_batch = batch["damaged"].to(device)
            clean_batch = batch["clean"].to(device)

            # Generate restored images
            generated_batch = generator(damaged_batch)

            # Loop through samples inside the batch
            for i in range(damaged_batch.size(0)):

                # Stop if maximum sample limit is reached
                if args.max_samples is not None and sample_index >= args.max_samples:
                    break

                # Convert tensors to uint8 RGB images
                damaged_image = tensor_to_uint8_image(damaged_batch[i])
                clean_image = tensor_to_uint8_image(clean_batch[i])
                generated_image = tensor_to_uint8_image(generated_batch[i])

                # Calculate PSNR
                psnr_score = calculate_psnr(
                    clean_image=clean_image,
                    generated_image=generated_image
                )

                # Calculate SSIM
                ssim_score = calculate_ssim(
                    clean_image=clean_image,
                    generated_image=generated_image
                )

                # Compare classical document features
                feature_comparison = compare_document_features(
                    before_image=damaged_image,
                    after_image=generated_image
                )

                # Create metric row
                row = {
                    "sample_index": sample_index,
                    "psnr": psnr_score,
                    "ssim": ssim_score,
                    "sharpness_before": feature_comparison["before"]["sharpness"],
                    "sharpness_after": feature_comparison["after"]["sharpness"],
                    "contrast_before": feature_comparison["before"]["contrast"],
                    "contrast_after": feature_comparison["after"]["contrast"],
                    "edge_density_before": feature_comparison["before"]["edge_density"],
                    "edge_density_after": feature_comparison["after"]["edge_density"]
                }

                # Optional OCR evaluation
                if args.use_ocr:
                    ocr_metrics = calculate_ocr_metrics(
                        damaged_image=damaged_image,
                        generated_image=generated_image,
                        clean_image=clean_image
                    )

                    # Add OCR metrics to row
                    row.update(ocr_metrics)

                # Save generated and clean images for optional FID
                generated_path = generated_dir / f"generated_{sample_index:04d}.png"
                clean_path = clean_dir / f"clean_{sample_index:04d}.png"

                save_image(generated_image, generated_path)
                save_image(clean_image, clean_path)

                # Save qualitative comparison for first N samples
                if sample_index < args.num_qualitative:
                    comparison_path = qualitative_dir / f"comparison_{sample_index:04d}.png"

                    save_qualitative_comparison(
                        damaged_image=damaged_image,
                        generated_image=generated_image,
                        clean_image=clean_image,
                        output_path=comparison_path
                    )

                # Add row to metric list
                metric_rows.append(row)

                # Increase sample index
                sample_index += 1

            # Stop outer loop if maximum sample limit is reached
            if args.max_samples is not None and sample_index >= args.max_samples:
                break

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metric_rows)

    # Save metrics CSV
    metrics_csv_path = metrics_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save metric plots
    plot_metric_values(
        metrics_df=metrics_df,
        output_dir=plots_dir
    )

    # Compute optional FID
    fid_score = None

    if args.compute_fid:
        fid_score = compute_optional_fid(
            real_dir=clean_dir,
            generated_dir=generated_dir
        )

    # Save summary report
    summary_path = metrics_dir / "evaluation_summary.txt"

    write_summary_report(
        metrics_df=metrics_df,
        output_path=summary_path,
        fid_score=fid_score
    )

    # Print final results
    print("Evaluation completed successfully.")
    print(f"Metrics saved at: {metrics_csv_path}")
    print(f"Summary saved at: {summary_path}")
    print(f"Plots saved at: {plots_dir}")
    print(f"Qualitative examples saved at: {qualitative_dir}")


def parse_args():
    """
    Defines command-line arguments for evaluation.
    """

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate DocRestoreAI document restoration model"
    )

    # Folder containing clean target images
    parser.add_argument(
        "--clean_dir",
        type=str,
        default="data/clean",
        help="Folder containing clean document images"
    )

    # Optional folder containing damaged images
    parser.add_argument(
        "--damaged_dir",
        type=str,
        default=None,
        help="Optional folder containing damaged document images"
    )

    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/final_model.pth",
        help="Path to trained model checkpoint"
    )

    # Output folder
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Folder where evaluation outputs will be saved"
    )

    # Image size
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size used by the model"
    )

    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )

    # Number of CPU workers
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers"
    )

    # Generate synthetic damage
    parser.add_argument(
        "--generate_damage",
        action="store_true",
        default=True,
        help="Generate damaged images automatically"
    )

    # Maximum number of samples to evaluate
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate"
    )

    # Number of qualitative examples to save
    parser.add_argument(
        "--num_qualitative",
        type=int,
        default=10,
        help="Number of qualitative comparison images to save"
    )

    # Optional OCR evaluation
    parser.add_argument(
        "--use_ocr",
        action="store_true",
        help="Use OCR readability evaluation"
    )

    # Optional FID calculation
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        help="Compute FID score using clean-fid"
    )

    return parser.parse_args()


if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_args()

    # Run evaluation
    evaluate_model(args)