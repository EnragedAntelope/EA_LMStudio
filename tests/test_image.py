"""Tests for lms_image: ComfyUI tensor -> PIL conversion (incl. RGBA fix)."""
from io import BytesIO

import numpy as np

from lms_image import convert_image_to_pil, resize_image
from PIL import Image


class FakeTensor:
    """Minimal stand-in for a torch tensor: exposes shape, indexing, cpu().numpy()."""

    def __init__(self, arr):
        self._arr = arr

    @property
    def shape(self):
        return self._arr.shape

    def __getitem__(self, idx):
        return FakeTensor(self._arr[idx])

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _tensor(h, w, c):
    return FakeTensor(np.linspace(0, 1, h * w * c, dtype=np.float32).reshape(1, h, w, c))


def test_rgb_tensor_stays_rgb():
    pil = convert_image_to_pil(_tensor(8, 8, 3))
    assert pil is not None
    assert pil.mode == "RGB"


def test_rgba_tensor_converted_and_jpeg_saveable():
    # The bug: a 4-channel (RGBA) tensor used to crash at JPEG save time.
    pil = convert_image_to_pil(_tensor(8, 8, 4))
    assert pil is not None
    assert pil.mode == "RGB"
    # Must not raise "cannot write mode RGBA as JPEG".
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    assert buf.getbuffer().nbytes > 0


def test_single_channel_tensor_is_jpeg_saveable():
    pil = convert_image_to_pil(_tensor(8, 8, 1))
    assert pil is not None
    assert pil.mode in ("L", "RGB")
    buf = BytesIO()
    pil.save(buf, format="JPEG")
    assert buf.getbuffer().nbytes > 0


def test_none_returns_none():
    assert convert_image_to_pil(None) is None


def test_out_of_range_values_clipped_not_wrapped():
    # 1.02 * 255 would wrap to a small value on a raw uint8 cast; clip prevents it.
    arr = np.full((1, 4, 4, 3), 1.02, dtype=np.float32)
    pil = convert_image_to_pil(FakeTensor(arr))
    assert pil is not None
    assert np.asarray(pil).max() == 255


def test_resize_downscales_longest_edge():
    pil = Image.new("RGB", (2000, 1000))
    out = resize_image(pil, 1000)
    assert max(out.size) == 1000
    assert out.size == (1000, 500)


def test_resize_noop_when_smaller():
    pil = Image.new("RGB", (100, 50))
    out = resize_image(pil, 512)
    assert out.size == (100, 50)


def test_convert_applies_resize():
    pil = convert_image_to_pil(_tensor(2000, 1000, 3), max_dimension=512)
    assert pil is not None
    assert max(pil.size) == 512
