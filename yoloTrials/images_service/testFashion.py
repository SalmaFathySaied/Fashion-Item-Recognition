
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2

# 1. Paths
image_path = Path("data/shirt.jpg")
model_path = Path("models/best.pt")
output_dir = Path("crops_fashion")
debug_dir = Path("debug")

output_dir.mkdir(parents=True, exist_ok=True)
debug_dir.mkdir(parents=True, exist_ok=True)

# 2. Load model and image
model = YOLO(str(model_path))
image = Image.open(image_path).convert("RGB")
image_width, image_height = image.size

print("Model classes:", model.names)
print("Original image size:", image.size)

# 3. Run detection
results = model.predict(
    source=str(image_path),
    conf=0.25
)

result = results[0]

# Save an image showing ALL predicted boxes
annotated = result.plot()
cv2.imwrite(
    str(debug_dir / "detections.jpg"),
    annotated
)

saved_count = 0

# 4. Process each detection
for i, box in enumerate(result.boxes):
    class_id = int(box.cls[0])
    class_name = result.names[class_id]
    confidence = float(box.conf[0])

    # Original predicted bounding box
    x1, y1, x2, y2 = map(
        int, box.xyxy[0].tolist()
    )

    width = x2 - x1
    height = y2 - y1

    print(
        f"Detection {i}: {class_name}, "
        f"confidence={confidence:.2f}, "
        f"box=({x1}, {y1}, {x2}, {y2}), "
        f"size={width}x{height}"
    )

    # Confidence filter
    if confidence < 0.50:
        print("Skipped: confidence below 0.50")
        continue

    # Minimum crop size
    if width < 32 or height < 32:
        print("Skipped: crop is too small")
        continue

    # 5. Add 5% padding around the predicted box
    pad_x = int(width * 0.05)
    pad_y = int(height * 0.05)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image_width, x2 + pad_x)
    y2 = min(image_height, y2 + pad_y)

    # 6. Crop from the original image
    crop = image.crop((x1, y1, x2, y2))

    filename = f"{i}_{class_name}_{confidence:.2f}.jpg"
    save_path = output_dir / filename
    crop.save(save_path)

    print(f"Saved: {save_path}")
    print(f"Final crop size: {crop.size}")
    print("-" * 30)

    saved_count += 1

print(f"Finished! Saved {saved_count} crops.")
print(f"Check bounding boxes: {debug_dir / 'detections.jpg'}")
