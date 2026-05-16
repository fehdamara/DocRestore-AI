"""
preprocessing.py

This file contains all preprocessing and post-processing functions used in the project.

Main responsibilities:
1. Load document images from disk.
2. Resize images to a fixed size.
3. Normalize images for deep learning.
4. Convert images between RGB, grayscale, NumPy arrays, and tensors.
5. Apply classical Computer Vision enhancement techniques.
6. Prepare restored images for saving or PDF reconstruction.

The code is heavily commented because the project must be easy to understand
and ready for presentation.
"""

# OpenCV is used for image processing operations
import cv2

# NumPy is used for numerical operations on images
import numpy as np

# Path is used to handle file paths in a clean and operating-system-independent way
from pathlib import Path


def read_image(image_path):
    """
    Reads an image from disk and converts it from BGR to RGB.

    OpenCV loads images in BGR format by default.
    Most deep learning libraries and plotting libraries use RGB format.
    Therefore, we convert BGR to RGB immediately after loading.

    Parameters:
        image_path: path to the image file

    Returns:
        image_rgb: image as a NumPy array in RGB format
    """

    # Convert the image path to a string because OpenCV expects a string path
    image_path = str(image_path)

    # Read the image using OpenCV
    image_bgr = cv2.imread(image_path)

    # If OpenCV cannot read the image, raise an error with a clear message
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert the image from BGR color format to RGB color format
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def save_image(image_rgb, output_path):
    """
    Saves an RGB image to disk.

    OpenCV saves images in BGR format, so before saving we convert the image
    from RGB back to BGR.

    Parameters:
        image_rgb: image as a NumPy array in RGB format
        output_path: destination path where the image will be saved
    """

    # Convert output path to Path object
    output_path = Path(output_path)

    # Create the parent folder if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB image to BGR because OpenCV expects BGR when saving
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Save the image to disk
    cv2.imwrite(str(output_path), image_bgr)


def resize_image(image, size=(256, 256)):
    """
    Resizes an image to a fixed size.

    Neural networks need images with the same dimensions.
    For this project we use 256x256 by default.

    Parameters:
        image: input image as a NumPy array
        size: target size as (width, height)

    Returns:
        resized_image: resized image
    """

    # Resize the image using OpenCV
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    return resized_image


def normalize_image(image):
    """
    Normalizes an image from range [0, 255] to range [-1, 1].

    GANs often work better when images are normalized between -1 and 1.
    This is also compatible with the final tanh activation often used in generators.

    Parameters:
        image: input image with pixel values between 0 and 255

    Returns:
        normalized_image: image with values between -1 and 1
    """

    # Convert image to float32 for numerical stability
    image = image.astype(np.float32)

    # Scale image from [0, 255] to [0, 1]
    image = image / 255.0

    # Scale image from [0, 1] to [-1, 1]
    normalized_image = (image * 2.0) - 1.0

    return normalized_image


def denormalize_image(image):
    """
    Converts an image from range [-1, 1] back to range [0, 255].

    This function is used after model prediction, before saving or displaying
    the restored image.

    Parameters:
        image: normalized image with values between -1 and 1

    Returns:
        output_image: uint8 image with values between 0 and 255
    """

    # Convert from [-1, 1] to [0, 1]
    image = (image + 1.0) / 2.0

    # Convert from [0, 1] to [0, 255]
    image = image * 255.0

    # Clip values to avoid invalid pixel values
    image = np.clip(image, 0, 255)

    # Convert to uint8, the standard image format
    output_image = image.astype(np.uint8)

    return output_image


def rgb_to_grayscale_3channel(image):
    """
    Converts an RGB image to grayscale but keeps 3 channels.

    Some neural networks expect 3-channel images.
    This function creates a grayscale image and duplicates it into 3 channels.

    Parameters:
        image: RGB image

    Returns:
        gray_3ch: grayscale image with 3 channels
    """

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert the single grayscale channel back to 3 channels
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return gray_3ch


def image_to_tensor_format(image):
    """
    Converts image format from HWC to CHW.

    OpenCV and NumPy store images as:
        height x width x channels

    PyTorch expects images as:
        channels x height x width

    Parameters:
        image: image in HWC format

    Returns:
        tensor_image: image in CHW format
    """

    # Transpose image dimensions from HWC to CHW
    tensor_image = np.transpose(image, (2, 0, 1))

    return tensor_image


def tensor_to_image_format(tensor_image):
    """
    Converts image format from CHW back to HWC.

    This is useful when converting a PyTorch model output back into an image.

    Parameters:
        tensor_image: image in CHW format

    Returns:
        image: image in HWC format
    """

    # Transpose image dimensions from CHW to HWC
    image = np.transpose(tensor_image, (1, 2, 0))

    return image


def apply_adaptive_threshold(image):
    """
    Applies adaptive thresholding to improve text readability.

    Adaptive thresholding is useful for documents with uneven lighting.
    It calculates a threshold locally for different image regions.

    Parameters:
        image: RGB image

    Returns:
        threshold_rgb: thresholded image in RGB format
    """

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    # Convert single-channel threshold image back to RGB
    threshold_rgb = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)

    return threshold_rgb


def apply_denoising(image):
    """
    Removes noise from a document image using OpenCV denoising.

    This is useful for scanned documents that contain small visual noise.

    Parameters:
        image: RGB image

    Returns:
        denoised_image: image after noise reduction
    """

    # Convert RGB to BGR because OpenCV denoising expects BGR-style images
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Apply Non-local Means Denoising
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Convert the denoised image back to RGB
    denoised_image = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

    return denoised_image


def enhance_contrast(image):
    """
    Enhances contrast using CLAHE.

    CLAHE means Contrast Limited Adaptive Histogram Equalization.
    It improves local contrast and is useful for faded documents.

    Parameters:
        image: RGB image

    Returns:
        enhanced_rgb: contrast-enhanced image
    """

    # Convert image from RGB to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Apply CLAHE only to the lightness channel
    enhanced_l = clahe.apply(l_channel)

    # Merge the enhanced lightness channel with the original color channels
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    # Convert LAB image back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return enhanced_rgb


def sharpen_image(image):
    """
    Applies a sharpening filter to make text edges clearer.

    Parameters:
        image: RGB image

    Returns:
        sharpened_image: sharpened RGB image
    """

    # Define a sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # Apply the sharpening kernel
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image


def deskew_image(image):
    """
    Attempts to correct small rotations in a document image.

    This function estimates the angle of the text/document and rotates it back.
    It is useful when a scanned document is slightly tilted.

    Parameters:
        image: RGB image

    Returns:
        deskewed_image: corrected image
    """

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invert the image because text is usually dark on bright background
    inverted = cv2.bitwise_not(gray)

    # Threshold the image to separate text from background
    _, threshold = cv2.threshold(
        inverted,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Find coordinates of all non-zero pixels
    coordinates = np.column_stack(np.where(threshold > 0))

    # If no text is detected, return original image
    if len(coordinates) == 0:
        return image

    # Estimate the minimum-area rectangle around the text pixels
    angle = cv2.minAreaRect(coordinates)[-1]

    # Correct the angle returned by OpenCV
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # If the angle is too large, avoid applying a wrong correction
    if abs(angle) > 15:
        return image

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate image center
    center = (width // 2, height // 2)

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image
    deskewed_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return deskewed_image


def classical_document_enhancement(image):
    """
    Applies a full classical Computer Vision enhancement pipeline.

    This function is useful as:
    1. A baseline method before deep learning.
    2. A post-processing method after the GAN output.
    3. A demonstration of classical CV techniques in the project.

    Steps:
        - Denoising
        - Contrast enhancement
        - Deskew correction
        - Sharpening

    Parameters:
        image: RGB image

    Returns:
        enhanced_image: enhanced document image
    """

    # Remove visual noise
    enhanced_image = apply_denoising(image)

    # Improve local contrast
    enhanced_image = enhance_contrast(enhanced_image)

    # Correct small rotation
    enhanced_image = deskew_image(enhanced_image)

    # Make text edges sharper
    enhanced_image = sharpen_image(enhanced_image)

    return enhanced_image