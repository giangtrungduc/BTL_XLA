"""
Module core/window.py
Tiện ích cửa sổ trượt (Sliding Window View) dùng chung cho các bộ lọc không gian (Chương 2).
Tận dụng stride_tricks của NumPy để vector hóa, đạt tốc độ tính toán cao
nhưng tuân thủ 100% công thức toán học trong bài giảng.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from core.padding import pad_image


def get_sliding_windows(channel: np.ndarray, kernel_size: int, pad_mode: str = "mirror") -> np.ndarray:
    """
    Tạo các cửa sổ trượt kích thước k x k cho ảnh 1 kênh 2D.
    
    Tham số:
        channel: Mảng 2D (H, W).
        kernel_size: Kích thước cửa sổ (phải là số lẻ: 3, 5, 7, ...).
        pad_mode: 'mirror', 'replicate', 'zero'.
        
    Trả về:
        Mảng view 4D có shape (H, W, kernel_size, kernel_size).
    """
    assert kernel_size % 2 == 1, "Kích thước cửa sổ phải là số lẻ (3, 5, 7, ...)"
    pad_len = kernel_size // 2
    padded = pad_image(channel, pad_len, pad_len, mode=pad_mode)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return windows
