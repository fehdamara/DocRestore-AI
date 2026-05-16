"""
streamlit_app.py

This file creates a simple web application for DocRestoreAI.

The app allows the user to:
1. Upload a damaged document image.
2. Upload a scanned PDF.
3. Restore the document using the trained Generator model.
4. Optionally apply classical Computer Vision post-processing.
5. Download the restored image or PDF.

This makes the project easier to demonstrate during the oral exam.
"""

# sys is used to fix import paths when running Streamlit from the app folder
import sys

# tempfile is used to store uploaded files temporarily
import tempfile

# Path is used for clean file path handling
from pathlib import Path

# Streamlit is used to build the web interface
import streamlit as st

# PyTorch is used to load and run the trained model
import torch

# PIL is used to display images in Streamlit
from PIL import Image

# NumPy is used to convert images into arrays
import numpy as np


# ------------------------------------------------------------
# Fix Python path
# ------------------------------------------------------------

# This allows the app to import files from the src folder.
# Without this, Streamlit may not find the project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ------------------------------------------------------------
# Import project functions
# ------------------------------------------------------------

# Import the Generator model
from src.models import UNetGenerator

# Import inference functions
from src.inference import (
    restore_image_array,
    restore_pdf_file,
    is_pdf,
    is_image
)

# Import preprocessing functions
from src.preprocessing import classical_document_enhancement

# Import utility function to select CPU or GPU
from src.utils import get_device


# ------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="DocRestoreAI",
    page_icon="📄",
    layout="wide"
)


# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #f3f6ff;
        border-left: 5px solid #4a63ff;
        margin-bottom: 20px;
    }

    .warning-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #fff6e5;
        border-left: 5px solid #ffb000;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

@st.cache_resource
def load_generator_model(checkpoint_path):
    """
    Loads the trained Generator model.

    Streamlit uses cache_resource so the model is loaded only once.
    This makes the app faster after the first run.

    Parameters:
        checkpoint_path: path to the trained model checkpoint

    Returns:
        generator: trained Generator model
        device: cpu or cuda
    """

    # Select the best available device
    device = get_device()

    # Create the Generator architecture
    generator = UNetGenerator(
        in_channels=3,
        out_channels=3
    ).to(device)

    # Load the checkpoint file
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

    return generator, device


def pil_to_rgb_array(pil_image):
    """
    Converts a PIL image into an RGB NumPy array.

    Parameters:
        pil_image: image loaded with PIL

    Returns:
        image_rgb: RGB image as NumPy array
    """

    # Convert image to RGB mode
    pil_image = pil_image.convert("RGB")

    # Convert PIL image to NumPy array
    image_rgb = np.array(pil_image)

    return image_rgb


def rgb_array_to_pil(image_rgb):
    """
    Converts an RGB NumPy array into a PIL image.

    Parameters:
        image_rgb: RGB image as NumPy array

    Returns:
        pil_image: PIL image
    """

    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(image_rgb)

    return pil_image


def save_uploaded_file(uploaded_file, suffix):
    """
    Saves an uploaded Streamlit file into a temporary file.

    Parameters:
        uploaded_file: file uploaded through Streamlit
        suffix: file extension, for example ".png" or ".pdf"

    Returns:
        temp_path: path to the temporary saved file
    """

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    # Write uploaded bytes into the temporary file
    temp_file.write(uploaded_file.read())

    # Close file so other functions can access it
    temp_file.close()

    return Path(temp_file.name)


def create_image_download_bytes(image_rgb):
    """
    Converts a restored image into bytes for Streamlit download button.

    Parameters:
        image_rgb: restored RGB image

    Returns:
        image_bytes: image data in PNG format
    """

    # Convert NumPy image to PIL
    pil_image = rgb_array_to_pil(image_rgb)

    # Create temporary file in memory
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    # Save image as PNG
    pil_image.save(temp_file.name, format="PNG")

    # Read saved image bytes
    with open(temp_file.name, "rb") as file:
        image_bytes = file.read()

    return image_bytes


# ------------------------------------------------------------
# App header
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">DocRestoreAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Generative Document Restoration and PDF Enhancement using Image-to-Image Translation</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    Upload a damaged document image or scanned PDF. The system restores the document
    using a Pix2Pix-style Generator and optional classical Computer Vision post-processing.
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Sidebar settings
# ------------------------------------------------------------

st.sidebar.header("Settings")

# Path to trained model
checkpoint_path = st.sidebar.text_input(
    "Model checkpoint path",
    value="outputs/checkpoints/final_model.pth"
)

# Image size used by the model
image_size = st.sidebar.selectbox(
    "Model image size",
    options=[128, 256, 512],
    index=1
)

# Post-processing option
apply_postprocessing = st.sidebar.checkbox(
    "Apply classical CV post-processing",
    value=True
)

# Classical-only fallback option
use_classical_only = st.sidebar.checkbox(
    "Use classical enhancement only",
    value=False
)

st.sidebar.markdown("---")

st.sidebar.write("Supported files:")
st.sidebar.write("- PNG")
st.sidebar.write("- JPG / JPEG")
st.sidebar.write("- BMP")
st.sidebar.write("- TIFF")
st.sidebar.write("- PDF")


# ------------------------------------------------------------
# File uploader
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a damaged document image or scanned PDF",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"]
)


# ------------------------------------------------------------
# Main app logic
# ------------------------------------------------------------

if uploaded_file is not None:

    # Get uploaded file extension
    file_suffix = Path(uploaded_file.name).suffix.lower()

    # Save uploaded file temporarily
    input_path = save_uploaded_file(
        uploaded_file=uploaded_file,
        suffix=file_suffix
    )

    # --------------------------------------------------------
    # PDF processing
    # --------------------------------------------------------

    if is_pdf(input_path):

        st.subheader("PDF Restoration")

        st.write("Uploaded PDF:", uploaded_file.name)

        # Create temporary output PDF path
        output_pdf_path = Path(tempfile.NamedTemporaryFile(
            delete=False,
            suffix="_restored.pdf"
        ).name)

        # Button to start PDF restoration
        if st.button("Restore PDF"):

            with st.spinner("Restoring PDF pages..."):

                # If user selected classical-only mode, show warning
                if use_classical_only:
                    st.warning(
                        "Classical-only PDF mode is not implemented in this app. "
                        "Please use the trained model mode for PDF restoration."
                    )

                else:
                    # Check if model checkpoint exists
                    if not Path(checkpoint_path).exists():
                        st.error(
                            "Model checkpoint not found. Train the model first or check the checkpoint path."
                        )

                    else:
                        # Load trained Generator
                        generator, device = load_generator_model(checkpoint_path)

                        # Restore PDF
                        restore_pdf_file(
                            input_pdf_path=input_path,
                            output_pdf_path=output_pdf_path,
                            generator=generator,
                            device=device,
                            image_size=image_size,
                            apply_postprocessing=apply_postprocessing
                        )

                        # Read restored PDF bytes
                        with open(output_pdf_path, "rb") as file:
                            pdf_bytes = file.read()

                        st.success("PDF restoration completed.")

                        # Download restored PDF
                        st.download_button(
                            label="Download restored PDF",
                            data=pdf_bytes,
                            file_name="restored_document.pdf",
                            mime="application/pdf"
                        )

    # --------------------------------------------------------
    # Image processing
    # --------------------------------------------------------

    elif is_image(input_path):

        st.subheader("Image Restoration")

        # Load uploaded image using PIL
        pil_image = Image.open(input_path)

        # Convert image to RGB NumPy array
        input_image_rgb = pil_to_rgb_array(pil_image)

        # Display input image
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                input_image_rgb,
                caption="Damaged Input",
                use_container_width=True
            )

        # Button to restore image
        if st.button("Restore Image"):

            with st.spinner("Restoring image..."):

                # ------------------------------------------------
                # Classical-only mode
                # ------------------------------------------------

                if use_classical_only:

                    # Apply classical OpenCV enhancement only
                    restored_image = classical_document_enhancement(
                        input_image_rgb
                    )

                # ------------------------------------------------
                # Deep learning model mode
                # ------------------------------------------------

                else:
                    # Check if checkpoint exists
                    if not Path(checkpoint_path).exists():
                        st.error(
                            "Model checkpoint not found. Train the model first or check the checkpoint path."
                        )
                        st.stop()

                    # Load trained Generator
                    generator, device = load_generator_model(checkpoint_path)

                    # Restore image using the Generator
                    restored_image = restore_image_array(
                        image_rgb=input_image_rgb,
                        generator=generator,
                        device=device,
                        image_size=image_size,
                        apply_postprocessing=apply_postprocessing
                    )

                # Display restored image
                with col2:
                    st.image(
                        restored_image,
                        caption="Restored Output",
                        use_container_width=True
                    )

                # Prepare image for download
                image_bytes = create_image_download_bytes(restored_image)

                st.success("Image restoration completed.")

                # Download button
                st.download_button(
                    label="Download restored image",
                    data=image_bytes,
                    file_name="restored_document.png",
                    mime="image/png"
                )

    # --------------------------------------------------------
    # Unsupported file
    # --------------------------------------------------------

    else:
        st.error("Unsupported file type.")


else:
    st.markdown(
        """
        <div class="warning-box">
        Upload an image or PDF to start the restoration process.
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")

st.write(
    "Ethical note: This tool is designed for readability enhancement and document restoration. "
    "It should not be used to falsify, alter, or manipulate official document content."
)