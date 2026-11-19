"""
Module core/resize.py
Thu nhỏ ảnh bằng trung bình khối (block averaging) - dùng cho khung xem trước của giao diện.
Cài đặt bằng NumPy, không dùng thư viện xử lý ảnh.

Với hệ số thu nhỏ nguyên f, mỗi khối f x f điểm ảnh được thay bằng giá trị trung bình của khối.
Đây chính là lọc trung bình f x f rồi lấy mẫu lại, nên tránh được hiện tượng răng cưa (aliasing).
"""
import math
import numpy as np


def downscale_factor(shape, max_side: int) -> int:
    """Hệ số nguyên nhỏ nhất để cạnh dài nhất của ảnh không vượt quá max_side."""
    longest = max(shape[0], shape[1])
    return max(1, int(math.ceil(longest / float(max_side))))


def block_downscale(image: np.ndarray, factor: int) -> np.ndarray:
    """Thu nhỏ ảnh theo hệ số nguyên factor bằng trung bình khối factor x factor."""
    if factor <= 1:
        return image.copy()
    h, w = image.shape[:2]
    h2, w2 = h // factor, w // factor
    if h2 == 0 or w2 == 0:
        return image.copy()
    cropped = image[:h2 * factor, :w2 * factor].astype(np.float32)
    if image.ndim == 2:
        blocks = cropped.reshape(h2, factor, w2, factor)
        out = blocks.mean(axis=(1, 3))
    else:
        c = image.shape[2]
        blocks = cropped.reshape(h2, factor, w2, factor, c)
        out = blocks.mean(axis=(1, 3))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
