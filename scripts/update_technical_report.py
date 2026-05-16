"""
update_technical_report.py

This script updates the Technical Analysis Document with real evaluation results.

It reads:
    outputs/evaluation/metrics/evaluation_summary.txt

Then it updates:
    docs/technical_analysis.md

Finally, it regenerates:
    docs/technical_analysis.pdf
"""

from pathlib import Path
import subprocess
import sys
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


def read_evaluation_summary(summary_path):
    """
    Reads evaluation results from evaluation_summary.txt.
    """

    summary_path = Path(summary_path)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Evaluation summary not found: {summary_path}\n"
            "Run first: python -m src.evaluate --max_samples 5"
        )

    results = {
        "Average PSNR": "Not available",
        "Average SSIM": "Not available",
        "FID Score": "Not computed",
        "Average OCR Improvement": "Not available",
    }

    lines = summary_path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        line = line.strip()

        if line.startswith("Average PSNR:"):
            results["Average PSNR"] = line.replace("Average PSNR:", "").strip()

        elif line.startswith("Average SSIM:"):
            results["Average SSIM"] = line.replace("Average SSIM:", "").strip()

        elif line.startswith("FID Score:"):
            results["FID Score"] = line.replace("FID Score:", "").strip()

        elif line.startswith("Average OCR Improvement:"):
            results["Average OCR Improvement"] = line.replace(
                "Average OCR Improvement:",
                ""
            ).strip()

    return results


def build_quantitative_results_section(results):
    """
    Builds the full markdown section for quantitative results.
    """

    section = f"""
## 3.1 Quantitative Results

After running the evaluation script, the results are saved in:

- outputs/evaluation/metrics/evaluation_summary.txt
- outputs/evaluation/metrics/evaluation_metrics.csv

The following table shows the experimental results obtained from the evaluation pipeline:

| Metric | Result |
|---|---|
| Average PSNR | {results["Average PSNR"]} |
| Average SSIM | {results["Average SSIM"]} |
| FID Score | {results["FID Score"]} |
| Average OCR Improvement | {results["Average OCR Improvement"]} |

The evaluation also generates visual plots and qualitative comparison images.

Generated files:

- outputs/evaluation/plots/psnr_plot.png
- outputs/evaluation/plots/ssim_plot.png
- outputs/evaluation/qualitative/

These files are used to support the experimental results section.

"""
    return section


def update_markdown_report(markdown_path, results):
    """
    Replaces the quantitative results section in the markdown report.
    """

    markdown_path = Path(markdown_path)

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown report not found: {markdown_path}")

    content = markdown_path.read_text(encoding="utf-8")

    start_marker = "## 3.1 Quantitative Results"
    end_marker = "## 3.2 Qualitative Results"

    new_section = build_quantitative_results_section(results)

    if start_marker not in content:
        raise ValueError("Could not find section: ## 3.1 Quantitative Results")

    if end_marker not in content:
        raise ValueError("Could not find section: ## 3.2 Qualitative Results")

    before_section = content.split(start_marker)[0]
    after_section = content.split(end_marker)[1]

    updated_content = before_section + new_section + end_marker + after_section

    markdown_path.write_text(updated_content, encoding="utf-8")

    print(f"[OK] Updated markdown report: {markdown_path}")


def regenerate_pdf():
    """
    Regenerates the final Technical Analysis PDF.
    """

    pdf_script = Path("docs/create_technical_pdf.py")

    if not pdf_script.exists():
        raise FileNotFoundError("Missing docs/create_technical_pdf.py")

    print("[INFO] Regenerating PDF...")

    result = subprocess.run(
        [sys.executable, str(pdf_script)]
    )

    if result.returncode != 0:
        raise RuntimeError("PDF generation failed.")

    print("[OK] PDF regenerated: docs/technical_analysis.pdf")


def main():
    """
    Main function.
    """

    print("Updating Technical Analysis Document")
    print("===================================")

    summary_path = Path("outputs/evaluation/metrics/evaluation_summary.txt")
    markdown_path = Path("docs/technical_analysis.md")

    results = read_evaluation_summary(summary_path)

    print("\nParsed evaluation results:")
    for metric_name, metric_value in results.items():
        print(f"- {metric_name}: {metric_value}")

    print()

    update_markdown_report(
        markdown_path=markdown_path,
        results=results
    )

    regenerate_pdf()

    print("\nDone.")
    print("Updated files:")
    print("- docs/technical_analysis.md")
    print("- docs/technical_analysis.pdf")


if __name__ == "__main__":
    main()