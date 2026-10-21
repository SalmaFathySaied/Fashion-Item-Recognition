
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Paths
BASE_DIR = Path(__file__).resolve().parent
CATALOG_DIR = BASE_DIR / "catalog"
QUERY_PATH = BASE_DIR / "data" / "input" / "quality_test" / "query.jpg"

# Pretrained CLIP model
MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading CLIP model...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {key: value.to(device)
              for key, value in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)

        # Handle different Transformers output formats
        if hasattr(features, "pooler_output"):
            features = features.pooler_output

        embedding = features / features.norm(
            p=2, dim=-1, keepdim=True
        )

    return embedding


# Validate input paths
if not QUERY_PATH.exists():
    raise FileNotFoundError(f"Query image not found: {QUERY_PATH}")

if not CATALOG_DIR.exists():
    raise FileNotFoundError(f"Catalog folder not found: {CATALOG_DIR}")

catalog_images = [
    path for path in CATALOG_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

if not catalog_images:
    raise ValueError("Add product images to the catalog folder.")

# Encode query
print("Encoding query image...")
query_embedding = get_image_embedding(QUERY_PATH)

# Compare query with catalog
results = []

for image_path in catalog_images:
    try:
        catalog_embedding = get_image_embedding(image_path)

        similarity = torch.nn.functional.cosine_similarity(
            query_embedding, catalog_embedding
        ).item()

        results.append((image_path.name, similarity))

    except Exception as error:
        print(f"Skipped {image_path.name}: {error}")

# Highest similarity first
results.sort(key=lambda item: item[1], reverse=True)

print("\nTop matching products:")
for rank, (name, score) in enumerate(results[:5], start=1):
    print(f"{rank}. {name} | Similarity: {score:.4f}")

