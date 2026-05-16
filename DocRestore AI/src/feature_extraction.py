"""
feature_extraction.py

This file contains classical Computer Vision feature extraction methods.

Even if the main model is deep learning-based, this file is useful because
the project guideline requires feature engineering or feature representation.

In this project, we include:

1. HOG features
2. Canny edge features
3. Contour-based features
4. Brightness, contrast, and sharpness statistics

These features help us explain how document quality changes before and after restoration.
"""

# OpenCV is used for classical image processing
import cv2

# NumPy is used for numerical calculations
import numpy as np

# JSON is used to save feature reports
import json

# Path is used to manage file paths
from pathlib import Path

# HOG is a classical feature descriptor used for shape and edge information
from skimage.feature import hog

# Import image reading function from our preprocessing file
from src.preprocessing import read_image


def extract_hog_features(image_rgb, resize_size=(128, 128)):
    """
    Extracts HOG features from an RGB image.

    HOG means Histogram of Oriented Gradients.
    It describes the shape and edge structure of an image.

    For document restoration, HOG can help represent:
    - text edges
    - document borders
    - line structures

    Parameters:
        image_rgb: input RGB image
        resize_size: size used before extracting HOG

    Returns:
        hog_features: feature vector
        hog_image: visual representation of HOG features
    """

    # Resize image to make feature extraction consistent
    resized_image = cv2.resize(image_rgb, resize_size)

    # Convert RGB image to grayscale because HOG works on intensity gradients
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

    # Extract HOG features and also return a visualization image
    hog_features, hog_image = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True
    )

    return hog_features, hog_image


def extract_canny_edges(image_rgb, threshold1=100, threshold2=200):
    """
    Extracts Canny edges from a document image.

    Canny edge detection is useful for detecting:
    - text boundaries
    - page borders
    - strong document lines

    Parameters:
        image_rgb: input RGB image
        threshold1: lower threshold for Canny
        threshold2: upper threshold for Canny

    Returns:
        edges: binary edge image
        edge_density: percentage of pixels detected as edges
    """

    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, threshold1, threshold2)

    # Calculate edge density
    # This is the ratio of edge pixels to all pixels
    edge_density = np.sum(edges > 0) / edges.size

    return edges, edge_density


def count_contours(image_rgb):
    """
    Counts contours in a document image.

    Contours are useful for detecting connected components such as:
    - letters
    - words
    - document regions
    - page boundaries

    Parameters:
        image_rgb: input RGB image

    Returns:
        contour_count: number of detected contours
    """

    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to separate foreground from background
    _, threshold = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Count the number of contours
    contour_count = len(contours)

    return contour_count


def calculate_sharpness(image_rgb):
    """
    Calculates image sharpness using the variance of the Laplacian.

    A blurry image usually has a low Laplacian variance.
    A sharper image usually has a higher Laplacian variance.

    Parameters:
        image_rgb: input RGB image

    Returns:
        sharpness: numerical sharpness score
    """

    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Apply Laplacian filter and calculate variance
    sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    return float(sharpness)


def calculate_brightness(image_rgb):
    """
    Calculates average brightness of an image.

    Parameters:
        image_rgb: input RGB image

    Returns:
        brightness: average pixel intensity
    """

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Calculate mean intensity
    brightness = np.mean(gray_image)

    return float(brightness)


def calculate_contrast(image_rgb):
    """
    Calculates image contrast.

    We use the standard deviation of grayscale pixel values.
    Higher standard deviation usually means higher contrast.

    Parameters:
        image_rgb: input RGB image

    Returns:
        contrast: contrast score
    """

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Calculate standard deviation
    contrast = np.std(gray_image)

    return float(contrast)


def calculate_dark_pixel_ratio(image_rgb, threshold=80):
    """
    Calculates the ratio of dark pixels.

    In documents, dark pixels often correspond to text.
    This metric can help estimate text visibility.

    Parameters:
        image_rgb: input RGB image
        threshold: intensity threshold below which a pixel is considered dark

    Returns:
        dark_ratio: ratio of dark pixels
    """

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Count dark pixels
    dark_pixels = np.sum(gray_image < threshold)

    # Calculate ratio
    dark_ratio = dark_pixels / gray_image.size

    return float(dark_ratio)


def extract_document_features(image_rgb):
    """
    Extracts a complete set of classical document features.

    Parameters:
        image_rgb: input RGB image

    Returns:
        features: dictionary containing document quality features
    """

    # Extract Canny edges and edge density
    _, edge_density = extract_canny_edges(image_rgb)

    # Count contours
    contour_count = count_contours(image_rgb)

    # Calculate sharpness
    sharpness = calculate_sharpness(image_rgb)

    # Calculate brightness
    brightness = calculate_brightness(image_rgb)

    # Calculate contrast
    contrast = calculate_contrast(image_rgb)

    # Calculate dark pixel ratio
    dark_pixel_ratio = calculate_dark_pixel_ratio(image_rgb)

    # Store all features in a dictionary
    features = {
        "edge_density": edge_density,
        "contour_count": contour_count,
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "dark_pixel_ratio": dark_pixel_ratio
    }

    return features


def compare_document_features(before_image, after_image):
    """
    Compares document features before and after restoration.

    This helps us explain whether the restoration improved:
    - sharpness
    - contrast
    - text visibility
    - edge clarity

    Parameters:
        before_image: damaged input image
        after_image: restored output image

    Returns:
        comparison: dictionary with before, after, and difference values
    """

    # Extract features before restoration
    before_features = extract_document_features(before_image)

    # Extract features after restoration
    after_features = extract_document_features(after_image)

    # Create dictionary for comparison
    comparison = {
        "before": before_features,
        "after": after_features,
        "difference": {}
    }

    # Calculate feature differences
    for key in before_features:
        comparison["difference"][key] = after_features[key] - before_features[key]

    return comparison


def save_feature_report(feature_data, output_path):
    """
    Saves extracted feature data into a JSON file.

    Parameters:
        feature_data: dictionary containing features
        output_path: path where JSON report will be saved
    """

    # Convert output path to Path object
    output_path = Path(output_path)

    # Create output folder if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save dictionary as formatted JSON
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(feature_data, file, indent=4)


def extract_features_from_file(image_path, output_json_path=None):
    """
    Extracts document features from an image file.

    Parameters:
        image_path: path to input image
        output_json_path: optional path to save JSON features

    Returns:
        features: extracted feature dictionary
    """

    # Read image from disk
    image_rgb = read_image(image_path)

    # Extract document features
    features = extract_document_features(image_rgb)

    # Save JSON report if output path is provided
    if output_json_path is not None:
        save_feature_report(features, output_json_path)

    return features


if __name__ == "__main__":
    """
    Example usage:

    python -m src.feature_extraction
    """

    # Example image path
    example_path = "data/samples/test_document.png"

    # Extract features from example image
    features = extract_features_from_file(
        image_path=example_path,
        output_json_path="outputs/metrics/sample_features.json"
    )

    # Print extracted features
    print(features)