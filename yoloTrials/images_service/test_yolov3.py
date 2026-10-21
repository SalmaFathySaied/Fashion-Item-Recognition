
from ultralytics import YOLO
from pathlib import Path

# Load YOLOv3 pretrained on COCO
model = YOLO("yolov3u.pt")

project_dir = Path(__file__).resolve().parent
image_folder = project_dir / "test_images"
output_folder = project_dir / "yolov3_results"
output_folder.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".webp"}

print("Model classes:")
print(model.names)

for image_path in image_folder.iterdir():
    if image_path.suffix.lower() not in extensions:
        continue

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]
    save_path = output_folder / f"{image_path.stem}_result.jpg"
    result.save(filename=str(save_path))

    print(f"\nImage: {image_path.name}")

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected")
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])
        name = result.names[class_id]
        confidence = float(box.conf[0])
        print(f"{name}: {confidence:.2f}")

print("\nFinished! Check yolov3_results")
