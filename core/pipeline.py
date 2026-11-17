"""
Module core/pipeline.py
Quy trình khôi phục ảnh có thứ tự cố định và chẩn đoán tự động (Chương 0, 2, 3, 5, 6).
Toàn bộ thuật toán được cài đặt trực tiếp bằng NumPy (Code from scratch).

NGUYÊN TẮC THIẾT KẾ:
  1. Mỗi bước là một hàm thuần khiết: f(ảnh, tham số) -> ảnh mới.
  2. run_pipeline() LUÔN nhận ảnh gốc ban đầu và thực hiện lại tuần tự từ đầu theo 4 bước cố định:
        Khử nhiễu -> Vết xước -> Sáng & Tương phản -> Làm nét.
     Lý do: Đạo hàm khuếch đại nhiễu (Ch.6 tr.29-31), nên bắt buộc phải làm mịn trước khi làm nét.
  3. Cơ chế này loại bỏ hoàn toàn lỗi "dồn bước" (khi kéo slider không bị tích lũy kết quả cũ)
     và không cần quản lý ngăn xếp Undo/Redo phức tạp.
  4. Hỗ trợ cả ảnh màu (qua không gian HSI, chỉ xử lý kênh I theo Chương 3 tr.53).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

from core.denoise import (
    median_filter, arithmetic_mean_filter, geometric_mean_filter,
    harmonic_mean_filter, contraharmonic_mean_filter, max_filter, min_filter,
    midpoint_filter, alpha_trimmed_mean_filter, k_nearest_mean_filter,
    gaussian_filter
)
from core.scratch import detect_scratch_mask, fill_masked
from core.intensity import gamma_transform, log_transform, negative_transform, piecewise_linear_transform
from core.histogram import equalize_histogram, compute_stats
from core.sharpen import laplacian_sharpen, sobel_filter, combined_sharpen
from core.color import process_color_on_intensity


@dataclass
class Stage:
    name: str          # 'denoise', 'scratch', 'contrast', 'sharpen'
    enabled: bool = True
    method: str = "default"
    params: Dict[str, Any] = field(default_factory=dict)


def run_pipeline(orig_image: np.ndarray, stages: List[Stage]) -> np.ndarray:
    """
    Thực thi quy trình khôi phục ảnh tuần tự theo thứ tự cố định từ ảnh gốc.
    
    Tham số:
        orig_image: Ảnh gốc ban đầu (uint8, 2D hoặc 3D).
        stages: Danh sách các Stage điều khiển (denoise -> scratch -> contrast -> sharpen).
        
    Trả về:
        Ảnh kết quả sau khi qua các bước được kích hoạt.
    """
    if orig_image is None or orig_image.size == 0:
        return orig_image

    img = orig_image.copy()
    stage_map = {s.name: s for s in stages}

    # ========================================================
    # BƯỚC 1: KHỬ NHIỄU (Chương 2, slide 91-117, 133-141)
    # ========================================================
    s_denoise = stage_map.get("denoise")
    if s_denoise and s_denoise.enabled:
        method = s_denoise.method.lower()
        k_size = s_denoise.params.get("kernel_size", 3)
        pad_mode = s_denoise.params.get("pad_mode", "mirror")
        
        if "median" in method or "trung vị" in method:
            img = median_filter(img, kernel_size=k_size, pad_mode=pad_mode)
        elif "arithmetic" in method or "số học" in method or "mean" == method:
            img = arithmetic_mean_filter(img, kernel_size=k_size, pad_mode=pad_mode)
        elif "geometric" in method or "hình học" in method:
            img = geometric_mean_filter(img, kernel_size=k_size, pad_mode=pad_mode)
        elif "harmonic" in method and "contra" not in method and "ngược" not in method:
            img = harmonic_mean_filter(img, kernel_size=k_size, pad_mode=pad_mode)
        elif "contra" in method or "điều hòa ngược" in method:
            Q = s_denoise.params.get("Q", 1.5)
            img = contraharmonic_mean_filter(img, kernel_size=k_size, Q=Q, pad_mode=pad_mode)
        elif "max" in method or "cực đại" in method:
            img = max_filter(img, kernel_size=k_size, pad_mode=pad_mode)
        elif "min" in method or "cực tiểu" in method:
            img = min_filter(img, kernel_size=k_size, pad_mode=pad_mode)
        elif "midpoint" in method or "điểm giữa" in method:
            img = midpoint_filter(img, kernel_size=k_size, pad_mode=pad_mode)
        elif "alpha" in method or "cắt tỉa" in method:
            d = s_denoise.params.get("d", 2)
            img = alpha_trimmed_mean_filter(img, kernel_size=k_size, d=d, pad_mode=pad_mode)
        elif "k_nearest" in method or "gần nhất" in method:
            k = s_denoise.params.get("k", 5)
            img = k_nearest_mean_filter(img, kernel_size=k_size, k=k, pad_mode=pad_mode)
        elif "gauss" in method:
            sigma = s_denoise.params.get("sigma", 1.0)
            img = gaussian_filter(img, kernel_size=k_size, sigma=sigma, pad_mode=pad_mode)
        else:
            img = median_filter(img, kernel_size=k_size, pad_mode=pad_mode)

    # ========================================================
    # BƯỚC 2: GIẢM VẾT XƯỚC (Chương 6, 5, 2)
    # ========================================================
    s_scratch = stage_map.get("scratch")
    if s_scratch and s_scratch.enabled:
        sensitivity = s_scratch.params.get("sensitivity", 0.5)
        fill_method = s_scratch.params.get("fill_method", "median")
        kernel_size = s_scratch.params.get("kernel_size", 7)
        
        # 1. Phát hiện mặt nạ vết xước
        mask = detect_scratch_mask(img, sensitivity=sensitivity, method="median_diff")
        # 2. Điền vết xước bằng các điểm lành lặn lân cận ngoài mặt nạ
        if np.any(mask > 0):
            img = fill_masked(img, mask, method=fill_method, kernel_size=kernel_size)

    # ========================================================
    # BƯỚC 3: SÁNG VÀ TƯƠNG PHẢN (Chương 2, slide 43-78)
    # Ảnh màu được chuyển sang HSI và chỉ xử lý trên kênh I
    # ========================================================
    s_contrast = stage_map.get("contrast")
    if s_contrast and s_contrast.enabled:
        method = s_contrast.method.lower()
        
        def _apply_contrast(channel):
            if "equalize" in method or "cân bằng" in method:
                mode = s_contrast.params.get("eq_mode", "stretch_to_zero")
                return equalize_histogram(channel, method=mode)
            elif "gamma" in method:
                gamma = s_contrast.params.get("gamma", 0.85)
                return gamma_transform(channel, gamma=gamma)
            elif "log" in method:
                return log_transform(channel)
            elif "negative" in method or "âm bản" in method:
                return negative_transform(channel)
            elif "piecewise" in method or "từng đoạn" in method:
                r1 = s_contrast.params.get("r1", 50)
                s1 = s_contrast.params.get("s1", 20)
                r2 = s_contrast.params.get("r2", 200)
                s2 = s_contrast.params.get("s2", 230)
                return piecewise_linear_transform(channel, r1, s1, r2, s2)
            else:
                gamma = s_contrast.params.get("gamma", 0.85)
                return gamma_transform(channel, gamma=gamma)
                
        # Nếu là ảnh màu thì áp dụng quy trình HSI
        if img.ndim == 3:
            img = process_color_on_intensity(img, _apply_contrast)
        else:
            img = _apply_contrast(img)

    # ========================================================
    # BƯỚC 4: LÀM NÉT (Chương 2, slide 119-123; Chương 6, slide 20-28, 58-60)
    # ========================================================
    s_sharpen = stage_map.get("sharpen")
    if s_sharpen and s_sharpen.enabled:
        method = s_sharpen.method.lower()
        k = s_sharpen.params.get("k", 0.35)
        
        if "combined" in method or "kết hợp" in method:
            gamma_val = s_sharpen.params.get("gamma_val", 0.85)
            img = combined_sharpen(img, k_lap=k, gamma_val=gamma_val)
        elif "4" in method or "4-neighbor" in method:
            img = laplacian_sharpen(img, k=k, mode="4-neighbor", pad_mode="mirror")
        else:
            # Mặc định 8 lân cận
            img = laplacian_sharpen(img, k=k, mode="8-neighbor", pad_mode="mirror")

    return img


def diagnose(image: np.ndarray) -> List[Stage]:
    """
    Hàm chẩn đoán tự động tổn hại của ảnh (Chương 2, slide 58-62, 129).
    Dựa trên các luật đơn giản và ngưỡng thực nghiệm:
      1. Tỷ lệ điểm ảnh bằng 0 hoặc 255 cao (> 1.0%) -> Gợi ý nhiễu xung muối tiêu -> Bật Median 3x3.
      2. Độ lệch chuẩn (std) hẹp (< 40) hoặc dải histogram hẹp -> Bật Cân bằng Histogram.
      3. Mean quá tối (< 95) -> Bật Gamma < 1.0 (0.80).
         Mean quá sáng (> 165) -> Bật Gamma > 1.0 (1.25).
      4. Phát hiện vết xước/nứt trắng -> Bật Khử vết xước.
      5. Làm nét viền nhẹ (4 lân cận, k=0.30) để hoàn thiện chi tiết.
    """
    if image is None or image.size == 0:
        return [
            Stage("denoise", enabled=True, method="median", params={"kernel_size": 3}),
            Stage("scratch", enabled=False, method="median_fill", params={"sensitivity": 0.5}),
            Stage("contrast", enabled=True, method="gamma", params={"gamma": 0.85}),
            Stage("sharpen", enabled=True, method="laplacian_4", params={"k": 0.30})
        ]

    if image.ndim == 3:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        gray = image.astype(np.float64)

    mean_val, std_val = float(np.mean(gray)), float(np.std(gray))
    total_px = float(gray.size)

    # 1. Tỷ lệ điểm cực trị (0 hoặc 255)
    zeros_ratio = float(np.sum(gray <= 1)) / total_px
    white_ratio = float(np.sum(gray >= 254)) / total_px
    salt_pepper_ratio = zeros_ratio + white_ratio

    has_salt_pepper = (salt_pepper_ratio > 0.008)
    
    # 2. Kiểm tra vết xước (vùng sáng tương phản đột ngột)
    mask = detect_scratch_mask(image, sensitivity=0.5)
    scratch_ratio = float(np.sum(mask > 0)) / total_px
    has_scratches = (scratch_ratio > 0.005)

    # Khởi tạo các stage
    denoise_stage = Stage(
        name="denoise",
        enabled=True if has_salt_pepper else False,
        method="median",
        params={"kernel_size": 3, "pad_mode": "mirror"}
    )

    scratch_stage = Stage(
        name="scratch",
        enabled=True if has_scratches else False,
        method="median_fill",
        params={"sensitivity": 0.5, "fill_method": "median", "kernel_size": 7}
    )

    # Sáng và tương phản
    contrast_stage = Stage(name="contrast", enabled=True)
    if std_val < 38.0:
        # Tương phản quá thấp, dải histogram hẹp -> Cân bằng Histogram
        contrast_stage.method = "equalize_histogram"
        contrast_stage.params = {"eq_mode": "stretch_to_zero"}
    elif mean_val < 95.0:
        # Ảnh bị tối -> Gamma < 1 để kéo sáng
        contrast_stage.method = "gamma"
        contrast_stage.params = {"gamma": 0.80}
    elif mean_val > 165.0:
        # Ảnh bị bạc màu / chói -> Gamma > 1 để làm đậm
        contrast_stage.method = "gamma"
        contrast_stage.params = {"gamma": 1.25}
    else:
        # Ảnh vừa phải -> Gamma nhẹ nhàng
        contrast_stage.method = "gamma"
        contrast_stage.params = {"gamma": 0.88}

    # Làm nét
    sharpen_stage = Stage(
        name="sharpen",
        enabled=True,
        method="laplacian_4",
        params={"k": 0.30, "pad_mode": "mirror"}
    )

    return [denoise_stage, scratch_stage, contrast_stage, sharpen_stage]
