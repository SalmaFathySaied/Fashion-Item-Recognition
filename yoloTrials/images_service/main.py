
from ultralytics import YOLO
from pathlib import Path
import cv2

# 1. Define paths
image_path = Path("data/shirt.jpg")
output_dir = Path("crops")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Check input image
if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path.resolve()}"
    )

# 3. Load YOLOv8 Large model
model = YOLO("yolov8l.pt")

# 4. Detect objects
results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True
)

result = results[0]
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Could not read the image.")

height, width = image.shape[:2]
saved_count = 0

# 5. Crop each detected object
for index, box in enumerate(result.boxes):
    x1, y1, x2, y2 = map(
        int, box.xyxy[0].tolist()
    )

    # Keep coordinates inside image boundaries
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    # Skip invalid boxes
    if x2 <= x1 or y2 <= y1:
        continue

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    class_id = int(box.cls[0])
    class_name = result.names[class_id]
    confidence = float(box.conf[0])

    # Save the crop
    filename = (
        f"{index}_{class_name}_{confidence:.2f}.jpg"
    )
    save_path = output_dir / filename

    if cv2.imwrite(str(save_path), crop):
        saved_count += 1
        print(f"Saved: {save_path}")
        print(f"Class: {class_name}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 30)

print(f"Finished! Saved {saved_count} crops.")
