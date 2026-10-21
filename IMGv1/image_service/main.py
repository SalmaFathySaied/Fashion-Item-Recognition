
from pathlib import Path
from services.image_quality import assess_image_quality

test_folder = Path("data/input/quality_test")


    
for image_path in sorted(test_folder.iterdir()):
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue

    result = assess_image_quality(image_path)

    print(f"\nImage: {image_path.name}")

    if result["valid"]:
        print(f"Resolution: {result['width']} x {result['height']}")
        print(f"Brightness: {result['brightness']}")
        print(f"Contrast: {result['contrast']}")
        print(f"Sharpness score: {result['sharpness']}")
        print(f"Noise score: {result['noise_score']}")
        print(f"Warnings: {result['warnings']}")
    else:
        print(f"Errors: {result['errors']}")
        