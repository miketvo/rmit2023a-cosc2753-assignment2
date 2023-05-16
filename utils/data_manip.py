import numpy as np
from PIL import Image


def remove_transparency(image: Image) -> Image:
    if image.mode in ('RGBA', 'RGBa', 'LA', 'La', 'PA', 'P'):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        image = image.convert('RGB')
    return image


def resize_crop(image: Image, width: int, height: int) -> Image:
    original_aspect_ratio = image.width / image.height
    target_aspect_ratio = width / height

    if original_aspect_ratio > target_aspect_ratio:
        # Crop horizontally
        new_width = int(image.height * target_aspect_ratio)
        left = (image.width - new_width) // 2
        upper = 0
        right = left + new_width
        lower = image.height
    else:
        # Crop vertically
        new_height = int(image.width / target_aspect_ratio)
        left = 0
        upper = (image.height - new_height) // 2
        right = image.width
        lower = upper + new_height

    cropped_image = image.crop((left, upper, right, lower))
    resized_image = cropped_image.resize((width, height), Image.Resampling.LANCZOS)

    return resized_image


def normalize_pixels(image: Image) -> Image:
    image_array = np.array(image)
    normalized_image_array = image_array / 255.0  # Normalize pixel values to the range [0, 1]
    return Image.fromarray((normalized_image_array * 255).astype(np.uint8))

