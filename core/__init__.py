"""
Package core
Toàn bộ thuật toán xử lý ảnh số tự lập trình từ đầu bằng NumPy (Code from scratch).
Theo Chương 0 tr.8 bài giảng Xử lý ảnh (INT13146 - PTIT):
KHÔNG import cv2 trong package core.
"""
from core.padding import pad_image
from core.window import get_sliding_windows
from core.noise import add_gaussian_noise, add_uniform_noise, add_salt_and_pepper_noise
from core.denoise import (
    median_filter, arithmetic_mean_filter, geometric_mean_filter,
    harmonic_mean_filter, contraharmonic_mean_filter, max_filter, min_filter,
    midpoint_filter, alpha_trimmed_mean_filter, k_nearest_mean_filter,
    gaussian_filter, create_gaussian_kernel
)
from core.intensity import (
    gamma_transform, log_transform, negative_transform, piecewise_linear_transform
)
from core.histogram import compute_histogram, compute_stats, equalize_histogram
from core.color import rgb_to_hsi, hsi_to_rgb, process_color_on_intensity
from core.sharpen import (
    compute_laplacian, laplacian_sharpen, sobel_filter, combined_sharpen
)
from core.morphology import dilate, erode, morph_open, morph_close, connected_components
from core.scratch import otsu_threshold, detect_scratch_mask, fill_masked
from core.metrics import calculate_rms_error, calculate_snr_ms, calculate_psnr
from core.degrade import simulate_old_photo
from core.pipeline import Stage, run_pipeline, diagnose

__all__ = [
    # padding & window
    "pad_image",
    "get_sliding_windows",
    # noise
    "add_gaussian_noise",
    "add_uniform_noise",
    "add_salt_and_pepper_noise",
    # denoise
    "median_filter",
    "arithmetic_mean_filter",
    "geometric_mean_filter",
    "harmonic_mean_filter",
    "contraharmonic_mean_filter",
    "max_filter",
    "min_filter",
    "midpoint_filter",
    "alpha_trimmed_mean_filter",
    "k_nearest_mean_filter",
    "gaussian_filter",
    "create_gaussian_kernel",
    # intensity
    "gamma_transform",
    "log_transform",
    "negative_transform",
    "piecewise_linear_transform",
    # histogram
    "compute_histogram",
    "compute_stats",
    "equalize_histogram",
    # color
    "rgb_to_hsi",
    "hsi_to_rgb",
    "process_color_on_intensity",
    # sharpen
    "compute_laplacian",
    "laplacian_sharpen",
    "sobel_filter",
    "combined_sharpen",
    # morphology
    "dilate",
    "erode",
    "morph_open",
    "morph_close",
    "connected_components",
    # scratch
    "otsu_threshold",
    "detect_scratch_mask",
    "fill_masked",
    # metrics
    "calculate_rms_error",
    "calculate_snr_ms",
    "calculate_psnr",
    # degrade
    "simulate_old_photo",
    # pipeline
    "Stage",
    "run_pipeline",
    "diagnose",
]
