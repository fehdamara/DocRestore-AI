# DocRestoreAI: Generative Document Restoration and PDF Enhancement

## Project Overview

DocRestoreAI is a Computer Vision application designed to restore damaged, noisy, low-quality, or poorly scanned documents.

The system takes a damaged document image or scanned PDF as input and generates a cleaner, more readable version using an image-to-image translation pipeline.

The project combines classical Computer Vision techniques with a deep learning generative model based on Pix2Pix GAN.

```text
Input:  damaged / noisy / skewed document image or PDF
Output: restored / enhanced document image or PDF
```

---

## Real-World Problem

Many documents are scanned or photographed in poor conditions. This can create several problems:

- low contrast
- image noise
- blur
- shadows
- skewed pages
- poor text readability
- gray or dirty background
- black borders
- bad lighting

This project is useful for improving the readability of scanned documents, archived papers, school notes, office documents, and old PDF files.

The system is designed only for document restoration and readability enhancement. It must not be used to falsify, alter, or manipulate the semantic content of official documents.

---

## Main Features

- Restore damaged document images
- Restore scanned PDF files
- Generate synthetic damaged documents for training
- Use classical Computer Vision preprocessing
- Use Pix2Pix GAN for image-to-image document restoration
- Apply optional post-processing
- Evaluate the model with PSNR, SSIM, FID, OCR readability, and qualitative analysis
- Use a Streamlit web app for easy interaction

---

## Project Pipeline

The application follows a complete Computer Vision pipeline:

### 1. Data Acquisition and Preprocessing

Clean document images are collected and used as ground truth targets.

Artificially damaged versions are generated using:

- Gaussian noise
- blur
- rotation
- low contrast
- artificial shadows
- scanning-like degradation

The preprocessing stage includes:

- image loading
- resizing
- normalization
- RGB conversion
- PDF-to-image conversion
- data preparation for PyTorch

---

### 2. Feature Engineering and Representation

The project uses both classical and learned feature representations.

Classical Computer Vision features include:

- HOG features
- Canny edge detection
- contour detection
- brightness analysis
- contrast analysis
- sharpness analysis
- dark pixel ratio

The deep learning model learns features automatically using convolutional neural networks.

---

### 3. Core Logic: Generative Model

The main model is a Pix2Pix-style Conditional GAN.

It contains:

- Generator: U-Net architecture
- Discriminator: PatchGAN architecture

The Generator receives a damaged document image and tries to generate a clean restored version.

The Discriminator receives both the damaged image and the restored image, then decides whether the restored image looks real or fake.

---

### 4. Post-processing

After the model generates the restored image, optional post-processing is applied:

- denormalization
- pixel value clipping
- contrast enhancement
- sharpening
- denoising
- deskew correction
- PDF reconstruction

---

### 5. Performance Evaluation

The model is evaluated using metrics suitable for generative image restoration:

- PSNR
- SSIM
- FID
- OCR readability improvement
- qualitative visual comparison

The evaluation results are saved as:

- CSV tables
- summary text reports
- plots
- qualitative comparison images

---

## Model Architecture

### Generator

The Generator is based on U-Net.

U-Net is effective for image-to-image translation because it uses skip connections between encoder and decoder layers. These skip connections help preserve document structure, text edges, and layout.

### Discriminator

The Discriminator is based on PatchGAN.

PatchGAN evaluates small patches of the image instead of judging the whole image at once. This is useful for document restoration because local details such as text sharpness, noise, and background quality are important.

---

## Repository Structure

```text
docrestore-ai/
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── damaged/
│   └── samples/
│
├── src/
│   ├── preprocessing.py
│   ├── degradation.py
│   ├── dataset.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── pdf_utils.py
│   ├── utils.py
│   └── models/
│       ├── generator.py
│       └── discriminator.py
│
├── app/
│   └── streamlit_app.py
│
├── outputs/
│   ├── restored_images/
│   ├── restored_pdfs/
│   ├── plots/
│   ├── metrics/
│   ├── checkpoints/
│   └── generated_samples/
│
├── docs/
│   └── technical_analysis.pdf
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/docrestore-ai.git
cd docrestore-ai
```

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch is not installed correctly, install it manually:

```bash
python -m pip install torch torchvision torchaudio
```

---

## Dataset Preparation

Place clean document images inside:

```text
data/clean/
```

Example:

```text
data/clean/document_001.png
data/clean/document_002.png
data/clean/document_003.png
```

The project can automatically generate damaged versions during training.

Supported image formats:

- PNG
- JPG
- JPEG
- BMP
- TIFF

---

## Training

To train the Pix2Pix GAN:

```bash
python -m src.train --epochs 20 --batch_size 4
```

For a quick test:

```bash
python -m src.train --epochs 2 --batch_size 1
```

The trained model will be saved in:

```text
outputs/checkpoints/final_model.pth
```

Training samples will be saved in:

```text
outputs/generated_samples/
```

Training logs will be saved in:

```text
outputs/metrics/training_log.txt
```

---

## Inference

### Restore an image

```bash
python -m src.inference --input_path data/samples/test_document.png
```

### Restore an image with post-processing

```bash
python -m src.inference --input_path data/samples/test_document.png --postprocess
```

### Restore a PDF

```bash
python -m src.inference --input_path data/samples/test_scan.pdf --output_path outputs/restored_pdfs/restored_scan.pdf
```

---

## Evaluation

To evaluate the trained model:

```bash
python -m src.evaluate --max_samples 20
```

With OCR evaluation:

```bash
python -m src.evaluate --max_samples 20 --use_ocr
```

With FID calculation:

```bash
python -m src.evaluate --max_samples 20 --compute_fid
```

The evaluation output will be saved in:

```text
outputs/evaluation/
```

This folder contains:

```text
metrics/evaluation_metrics.csv
metrics/evaluation_summary.txt
plots/psnr_plot.png
plots/ssim_plot.png
qualitative/comparison_0000.png
generated_images/
clean_images/
```

---

## Streamlit Web App

To run the web application:

```bash
streamlit run app/streamlit_app.py
```

or:

```bash
python -m streamlit run app/streamlit_app.py
```

The app allows users to:

- upload damaged document images
- upload scanned PDFs
- restore documents using the trained model
- apply classical Computer Vision post-processing
- download restored images or PDFs

If the model is not trained yet, the app can still be tested with the classical enhancement option.

---

## Results Summary

The system is evaluated using quantitative and qualitative metrics.

Example result table:

| Metric          | Description                        | Expected Goal            |
| --------------- | ---------------------------------- | ------------------------ |
| PSNR            | Pixel-level reconstruction quality | Higher is better         |
| SSIM            | Structural similarity              | Higher is better         |
| FID             | Realism of generated images        | Lower is better          |
| OCR Improvement | Text readability improvement       | Positive value is better |

Example qualitative comparison:

```text
Damaged Input | Generated Restoration | Clean Target
```

The final Technical Analysis Document includes detailed results, tables, plots, and failure analysis.

---

## Failure Analysis

The model may fail in some situations, such as:

- extremely blurry documents
- very strong shadows
- handwriting with unusual style
- very low-resolution scans
- documents with complex backgrounds
- pages with heavy stains or missing content

Since document restoration is an image generation task, the model may sometimes generate visually plausible results that are not perfectly faithful to the original document.

---

## Ethical Considerations

This project is designed only for document restoration and readability improvement.

It should not be used to:

- falsify official documents
- modify signatures
- change dates
- alter names
- manipulate legal or administrative content
- create misleading documents

The restored output should be treated as an enhanced visual version, not as proof of original document authenticity.

---

## Technologies Used

- Python
- OpenCV
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-image
- scikit-learn
- PyMuPDF
- Streamlit
- pytesseract
- clean-fid

---

## Author

Project developed for the Computer Vision final project.

---

## License

This project is released for educational purposes.
