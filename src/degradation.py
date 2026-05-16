"""
degradation.py

This file contains functions used to artificially damage clean document images.
The goal is to create paired data for image-to-image translation:

Input:  damaged document image
Target: clean document image

These functions are useful because real paired datasets are hard to find.
By generating synthetic damage, we can train the model in a controlled way.
"""

# Importiamo OpenCV per modificare le immagini
import cv2

# Importiamo NumPy per lavorare con array e valori numerici
import numpy as np

# Importiamo random per applicare effetti casuali
import random


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Adds Gaussian noise to an image.

    Parameters:
        image: input image as a NumPy array
        mean: average value of the noise
        sigma: intensity of the noise

    Returns:
        noisy_image: image with added noise
    """

    # Creiamo rumore casuale con distribuzione gaussiana
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)

    # Convertiamo l'immagine in float per evitare problemi durante la somma
    image_float = image.astype(np.float32)

    # Aggiungiamo il rumore all'immagine originale
    noisy_image = image_float + noise

    # Limitiamo i valori tra 0 e 255 perché sono i valori validi per un'immagine
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convertiamo di nuovo l'immagine in uint8
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


def add_blur(image, kernel_size=5):
    """
    Adds blur to simulate a badly scanned or unfocused document.

    Parameters:
        image: input image
        kernel_size: size of the blur kernel

    Returns:
        blurred_image: blurred version of the image
    """

    # Il kernel size deve essere dispari, per esempio 3, 5, 7
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Applichiamo Gaussian Blur con OpenCV
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image


def rotate_image(image, angle=5):
    """
    Rotates the image to simulate a skewed scanned document.

    Parameters:
        image: input image
        angle: rotation angle in degrees

    Returns:
        rotated_image: rotated image
    """

    # Otteniamo altezza e larghezza dell'immagine
    height, width = image.shape[:2]

    # Calcoliamo il centro dell'immagine
    center = (width // 2, height // 2)

    # Creiamo la matrice di rotazione
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Applichiamo la rotazione
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated_image


def reduce_contrast(image, alpha=0.6, beta=30):
    """
    Reduces image contrast to simulate faded or poorly scanned documents.

    Parameters:
        image: input image
        alpha: contrast control
        beta: brightness control

    Returns:
        low_contrast_image: image with reduced contrast
    """

    # alpha controlla il contrasto, beta controlla la luminosità
    low_contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return low_contrast_image


def add_shadow(image):
    """
    Adds an artificial shadow to simulate a document photographed with bad lighting.

    Parameters:
        image: input image

    Returns:
        shadow_image: image with artificial shadow
    """

    # Otteniamo dimensioni immagine
    height, width = image.shape[:2]

    # Creiamo una maschera nera grande quanto l'immagine
    shadow_mask = np.zeros((height, width), dtype=np.uint8)

    # Scegliamo una posizione casuale per l'ombra
    start_x = random.randint(0, width // 2)
    end_x = random.randint(width // 2, width)

    # Creiamo un gradiente orizzontale per simulare l'ombra
    for x in range(start_x, end_x):
        intensity = int(120 * (x - start_x) / max(1, end_x - start_x))
        shadow_mask[:, x] = intensity

    # Convertiamo la maschera in 3 canali se l'immagine è RGB
    if len(image.shape) == 3:
        shadow_mask = cv2.merge([shadow_mask, shadow_mask, shadow_mask])

    # Sottraiamo l'ombra dall'immagine
    shadow_image = cv2.subtract(image, shadow_mask)

    return shadow_image


def damage_document(image):
    """
    Applies a random combination of degradation effects to a clean document image.

    Parameters:
        image: clean input document image

    Returns:
        damaged_image: artificially damaged document image
    """

    # Copiamo l'immagine originale per non modificarla direttamente
    damaged_image = image.copy()

    # Applichiamo rumore con una certa probabilità
    if random.random() < 0.7:
        damaged_image = add_gaussian_noise(damaged_image)

    # Applichiamo blur con una certa probabilità
    if random.random() < 0.5:
        damaged_image = add_blur(damaged_image, kernel_size=random.choice([3, 5, 7]))

    # Applichiamo rotazione casuale
    if random.random() < 0.6:
        angle = random.uniform(-7, 7)
        damaged_image = rotate_image(damaged_image, angle=angle)

    # Riduciamo contrasto
    if random.random() < 0.6:
        damaged_image = reduce_contrast(damaged_image)

    # Aggiungiamo ombra
    if random.random() < 0.5:
        damaged_image = add_shadow(damaged_image)

    return damaged_image