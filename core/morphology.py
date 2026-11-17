"""
Module core/morphology.py
Các phép toán hình thái học (Morphological Operations) - Chương 5, slide 23-70.
Cài đặt trực tiếp 100% bằng NumPy (Code from scratch), không dùng OpenCV.

Bao gồm:
  1. Dilation (Phép giãn) - Slide tr.27
  2. Erosion (Phép co / xói mòn) - Slide tr.25
  3. Opening (Phép mở: Co rồi Giãn) - Slide tr.35
  4. Closing (Phép đóng: Giãn rồi Co) - Slide tr.36
  5. Connected Components (Phân tích thành phần liên thông 8 lân cận) - Slide tr.64-70
"""
from typing import Tuple, List
import numpy as np
from core.window import get_sliding_windows


def dilate(image: np.ndarray, kernel_size: int = 3, pad_mode: str = "replicate") -> np.ndarray:
    """
    Phép giãn hình thái học (Dilation): f oplus b = max_{s,t in B} f(x-s, y-t) (Chương 5, slide 27).
    Làm nở các vùng sáng, hàn gắn các khe nứt nhỏ.
    """
    def _op(ch):
        win = get_sliding_windows(ch, kernel_size, pad_mode)
        return np.max(win, axis=(-2, -1)).astype(np.uint8)

    if image.ndim == 2:
        return _op(image)
    else:
        channels = [_op(image[..., c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)


def erode(image: np.ndarray, kernel_size: int = 3, pad_mode: str = "replicate") -> np.ndarray:
    """
    Phép co hình thái học (Erosion): f ominus b = min_{s,t in B} f(x+s, y+t) (Chương 5, slide 25).
    Làm thu nhỏ các vùng sáng, loại bỏ các chi tiết/vết xước sáng mảnh.
    """
    def _op(ch):
        win = get_sliding_windows(ch, kernel_size, pad_mode)
        return np.min(win, axis=(-2, -1)).astype(np.uint8)

    if image.ndim == 2:
        return _op(image)
    else:
        channels = [_op(image[..., c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)


def morph_open(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Phép mở hình thái học (Opening): f circ b = (f ominus b) oplus b (Chương 5, slide 35).
    Co trước rồi Giãn sau. Loại bỏ các chi tiết sáng nhỏ hơn phần tử cấu trúc.
    """
    return dilate(erode(image, kernel_size), kernel_size)


def morph_close(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Phép đóng hình thái học (Closing): f bullet b = (f oplus b) ominus b (Chương 5, slide 36).
    Giãn trước rồi Co sau. Lấp đầy các lỗ thủng tối và hàn gắn các vết nứt đứt đoạn.
    """
    return erode(dilate(image, kernel_size), kernel_size)


def connected_components(
    binary_mask: np.ndarray, 
    min_size: int = 5, 
    max_size: int = None
) -> Tuple[np.ndarray, int]:
    """
    Gán nhãn các thành phần liên thông 8 lân cận (8-Connected Components Labeling) - Chương 5, slide 64-70.
    Cài đặt thuần NumPy bằng thuật toán Flood-fill / BFS 8 lân cận.
    
    Tham số:
        binary_mask: Mảng boolean hoặc uint8 (0 và >0).
        min_size: Kích thước pixel tối thiểu để giữ lại (lọc bỏ nhiễu lốm đốm nhỏ).
        max_size: Kích thước pixel tối đa (lọc bỏ các mảng quá lớn không phải vết xước).
        
    Trả về:
        filtered_mask: Mặt nạ nhị phân uint8 (0 hoặc 255) chỉ giữ lại các thành phần hợp lệ.
        num_components: Số lượng thành phần liên thông tìm thấy.
    """
    h, w = binary_mask.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    is_foreground = (binary_mask > 0)
    
    filtered_mask = np.zeros((h, w), dtype=np.uint8)
    num_components = 0
    
    # 8 hướng lân cận
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # Tìm các điểm ảnh tiền cảnh
    y_indices, x_indices = np.where(is_foreground)
    
    for start_y, start_x in zip(y_indices, x_indices):
        if visited[start_y, start_x]:
            continue
            
        # Bắt đầu BFS cho 1 thành phần liên thông mới
        queue = [(start_y, start_x)]
        visited[start_y, start_x] = True
        component_pixels = [(start_y, start_x)]
        
        idx = 0
        while idx < len(queue):
            cy, cx = queue[idx]
            idx += 1
            
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if is_foreground[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
                        component_pixels.append((ny, nx))
                        
        size = len(component_pixels)
        if size >= min_size and (max_size is None or size <= max_size):
            num_components += 1
            for py, px in component_pixels:
                filtered_mask[py, px] = 255
                
    return filtered_mask, num_components
