
from pathlib import Path
import cv2


def assess_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        return {
            "valid": False,
            "reason": "File not found",
            "warnings": []
        }

    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "valid": False,
            "reason": "Invalid or unreadable image",
            "warnings": []
        }

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_score = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    warnings = []

    if width < 224 or height < 224:
        warnings.append("Low image resolution")

    # Preliminary warning thresholds only.
    # Calibrate these using labeled project images.
    if brightness < 40:
        warnings.append("Image may be too dark")
    elif brightness > 220:
        warnings.append("Image may be too bright")

    if blur_score < 100:
        warnings.append("Image may be blurry")

    return {
        "valid": True,
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2),
        "warnings": warnings,
        "quality_status": (
            "Review warnings" if warnings else "Passed basic checks"
        )
    }


if __name__ == "__main__":
    result = assess_image("data/shirt.jpg")

    for key, value in result.items():
        print(f"{key}: {value}")
