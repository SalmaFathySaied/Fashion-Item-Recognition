
from pathlib import Path
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_WIDTH = 32
MIN_HEIGHT = 32



def validate_image(image_path):
    image_path = Path(image_path)
    errors = []

    #Check lawt  file exists
    if not image_path.is_file():
        return {
            "valid": False,
            "errors": ["Image file does not exist."]
        }

    #Validate file extension
    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return {
            "valid": False,
            "errors": ["Unsupported file extension."]
        }

    #  Validate file size
    try:
        file_size = image_path.stat().st_size
    except OSError:
        return {
            "valid": False,
            "errors": ["Unable to access image file."]
        }

    if file_size == 0:
        return {
            "valid": False,
            "errors": ["Image file is empty."]
        }

    if file_size > MAX_FILE_SIZE:
        return {
            "valid": False,
            "errors": ["Image exceeds the 10 MB limit."]
        }

    # Validate image format and dimensions
    try:
        with Image.open(image_path) as img:
            image_format = img.format
            width, height = img.size

            if image_format not in SUPPORTED_FORMATS:
                return {
                    "valid": False,
                    "errors": ["Unsupported image format."]
                }

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                errors.append("Image dimensions are too small.")

           

            
            img.load()

            if errors:
                return {
                    "valid": False,
                    "format": image_format,
                    "width": width,
                    "height": height,
                    "file_size_bytes": file_size,
                    "errors": errors
                }

            return {
                "valid": True,
                "format": image_format,
                "width": width,
                "height": height,
                "file_size_bytes": file_size,
                "errors": []
            }

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
          Image.DecompressionBombError
    
    ):
        return {
            "valid": False,
            "errors": ["Invalid or corrupted image."]
        }
