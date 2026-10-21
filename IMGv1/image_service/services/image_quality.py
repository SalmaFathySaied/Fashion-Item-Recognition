
# Measure image quality and identify potential problems.

from pathlib import Path

import cv2
import numpy as np

from services.image_validator import validate_image


def assess_image_quality(img_path):
    # call function vlidationnn ll file de
    validation = validate_image(img_path)

    if not validation["valid"]:
        return {
            "valid": False,
            "errors": validation["errors"],
            "quality": None,
            "warnings": []
        }


    image = cv2.imread(str(img_path))

    if image is None:
        return {
            "valid": False,
            "errors": ["Cannot read image."],
            "quality": None,
            "warnings": []
        }

    #Get dimensions and convert to grayscale
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate quality metrics
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    # Estimate noise using a median-filter residual
    filtered = cv2.medianBlur(gray, 3)
    residual = (
        gray.astype(np.float32)
        - filtered.astype(np.float32)
    )
    noise_score = float(np.mean(np.abs(residual)))

    # Identify potential quality problems
    warnings = []
##DEEE ARKAM M4 SABTA HN8YRHA 2ODAM 3LA 7ASAB EL DATA MOHEMM
    # Resolution
    if width < 224 or height < 224:
        warnings.append("Low resolution")

    # Brightness
    if brightness < 50:
        warnings.append("Image may be too dark")
    elif brightness > 210:
        warnings.append("Image may be too bright")

    # Contrast
    if contrast < 15:
        warnings.append("Low contrast")

    # Sharpness / Blur
    if sharpness < 50:
        warnings.append("Possible blur")

    # Noise
    if noise_score > 8:
        warnings.append("Possible high noise")

    # 6. Return the results
    return {
        "valid": True,
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "noise_score": round(noise_score, 2),
        "warnings": warnings
    }
