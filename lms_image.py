"""
Image preprocessing for EA_LMStudio.

Converts ComfyUI IMAGE tensors into PIL images for the VLM path. Kept separate
from ``LMStudio.py`` (which imports the lmstudio SDK / ComfyUI and does a network
fetch at import time) so this pure numpy/Pillow logic stays unit-testable.

The tensor argument only needs ``.shape``, indexing, and ``.cpu().numpy()`` — i.e.
a torch tensor at runtime, but any object with that surface works.
"""
import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("EA_LMStudio")


def resize_image(pil_image: Image.Image, max_dimension: Optional[int]) -> Image.Image:
    """
    Resize image to fit within max_dimension while preserving aspect ratio.

    Args:
        pil_image: PIL Image to resize
        max_dimension: Maximum size for longest edge, or None to skip resize

    Returns:
        Resized PIL Image (or original if no resize needed)
    """
    if max_dimension is None:
        return pil_image

    width, height = pil_image.size
    max_current = max(width, height)

    # Only resize if image is larger than target
    if max_current <= max_dimension:
        return pil_image

    # Calculate new dimensions preserving aspect ratio
    scale = max_dimension / max_current
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Use LANCZOS for high-quality downscaling
    return pil_image.resize((new_width, new_height), Image.LANCZOS)


def convert_image_to_pil(image_tensor, max_dimension: Optional[int] = None) -> Optional[Image.Image]:
    """
    Convert a ComfyUI image tensor to a PIL Image, optionally resizing.

    Args:
        image_tensor: ComfyUI image tensor ([B,H,W,C] or [H,W,C], float 0-1)
        max_dimension: Max size for the longest edge, or None to skip resize

    Returns:
        PIL Image (always RGB or L mode, JPEG-safe) or None if conversion fails
    """
    try:
        # ComfyUI images are [B, H, W, C] float tensors in 0-1 range
        if image_tensor is None:
            return None

        # Take first image if batch
        if len(image_tensor.shape) == 4:
            img_array = image_tensor[0].cpu().numpy()
        else:
            img_array = image_tensor.cpu().numpy()

        # Convert to uint8. Clip first: ComfyUI tensors can slightly
        # exceed [0, 1] (VAE decode etc.), and out-of-range values would
        # otherwise wrap around during the uint8 cast (1.02 -> 4).
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

        # Drop a trailing singleton channel so a [H,W,1] array becomes a plain
        # [H,W] grayscale image; Image.fromarray can't handle a (H,W,1) shape.
        if img_array.ndim == 3 and img_array.shape[-1] == 1:
            img_array = img_array[..., 0]

        # Create PIL Image
        pil_image = Image.fromarray(img_array)

        # Normalize to a JPEG-encodable mode. ComfyUI IMAGE tensors are usually
        # [B,H,W,3], but 4-channel (RGBA) or single-channel tensors do occur;
        # JPEG can only encode RGB/L, so an RGBA image would raise "cannot write
        # mode RGBA as JPEG" at save time and lose the image. Converting here
        # keeps the VLM path robust to whatever channel count arrives.
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")

        # Apply resize if specified
        if max_dimension is not None:
            pil_image = resize_image(pil_image, max_dimension)

        return pil_image

    except Exception as e:
        logger.error(f"Failed to convert image: {e}")
        return None
