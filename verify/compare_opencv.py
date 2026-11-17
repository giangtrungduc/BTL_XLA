"""
Module verify/compare_opencv.py
Đối chứng nghiệm thu giữa thuật toán NumPy (tự lập trình từ đầu) và thư viện OpenCV C++ (Chương 0, slide 10).
THƯ VIỆN cv2 CHỈ ĐƯỢC PHÉP IMPORT Ở ĐÂY VÀ TRONG BÀI KIỂM THỬ ĐỐI CHỨNG.
"""
import time
from typing import Dict, Any, List
import numpy as np
import cv2

from core.denoise import median_filter, arithmetic_mean_filter, gaussian_filter
from core.histogram import equalize_histogram
from core.sharpen import laplacian_sharpen, LAPLACIAN_8_NEIGHBOR
from core.morphology import dilate, erode
from core.scratch import otsu_threshold
from core.metrics import calculate_rms_error, calculate_psnr


def compare_operation(image: np.ndarray, op_name: str, **kwargs) -> Dict[str, Any]:
    """
    So sánh đối chứng song song 1 phép toán giữa NumPy tự code và OpenCV.
    
    Tham số:
        image: Mảng numpy uint8 (ảnh xám).
        op_name: 'median', 'mean', 'gaussian', 'equalize_hist', 'laplacian', 'dilate', 'erode', 'otsu'.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    result = {
        "operation": op_name,
        "input_shape": gray.shape,
        "numpy_time_ms": 0.0,
        "opencv_time_ms": 0.0,
        "e_rms": 0.0,
        "psnr_db": 0.0,
        "numpy_output": None,
        "opencv_output": None,
        "passed": False,
        "note": ""
    }
    
    if op_name == "median":
        k_size = kwargs.get("kernel_size", 3)
        # cv2.medianBlur dùng mặc định BORDER_REPLICATE
        t0 = time.perf_counter()
        my_out = median_filter(gray, kernel_size=k_size, pad_mode="replicate")
        t1 = time.perf_counter()
        
        t2 = time.perf_counter()
        cv_out = cv2.medianBlur(gray, k_size)
        t3 = time.perf_counter()
        
    elif op_name == "mean":
        k_size = kwargs.get("kernel_size", 3)
        t0 = time.perf_counter()
        my_out = arithmetic_mean_filter(gray, kernel_size=k_size, pad_mode="mirror")
        t1 = time.perf_counter()
        
        t2 = time.perf_counter()
        cv_out = cv2.blur(gray, (k_size, k_size), borderType=cv2.BORDER_REFLECT_101)
        t3 = time.perf_counter()
        
    elif op_name == "gaussian":
        k_size = kwargs.get("kernel_size", 3)
        sigma = kwargs.get("sigma", 1.0)
        t0 = time.perf_counter()
        my_out = gaussian_filter(gray, kernel_size=k_size, sigma=sigma, pad_mode="mirror")
        t1 = time.perf_counter()
        
        t2 = time.perf_counter()
        cv_out = cv2.GaussianBlur(gray, (k_size, k_size), sigma, borderType=cv2.BORDER_REFLECT_101)
        t3 = time.perf_counter()
        
    elif op_name in ("equalize_hist", "histogram_equalization"):
        t0 = time.perf_counter()
        my_out = equalize_histogram(gray, method="stretch_to_zero")
        t1 = time.perf_counter()
        
        t2 = time.perf_counter()
        cv_out = cv2.equalizeHist(gray)
        t3 = time.perf_counter()
        
    elif op_name in ("laplacian", "sharpen"):
        k = kwargs.get("k", 0.4)
        t0 = time.perf_counter()
        my_out = laplacian_sharpen(gray, k=k, mode="8-neighbor", pad_mode="mirror")
        t1 = time.perf_counter()
        
        t2 = time.perf_counter()
        lap_cv = cv2.filter2D(gray.astype(np.float32), -1, LAPLACIAN_8_NEIGHBOR, borderType=cv2.BORDER_REFLECT_101)
        sharp_cv = gray.astype(np.float32) - k * lap_cv
        cv_out = np.clip(np.round(sharp_cv), 0, 255).astype(np.uint8)
        t3 = time.perf_counter()
        
    elif op_name == "dilate":
        k_size = kwargs.get("kernel_size", 3)
        kernel = np.ones((k_size, k_size), dtype=np.uint8)
        t0 = time.perf_counter()
        my_out = dilate(gray, kernel_size=k_size, pad_mode="replicate")
        t1 = time.perf_counter()
        
        t2 = time.perf_counter()
        cv_out = cv2.dilate(gray, kernel, borderType=cv2.BORDER_REPLICATE)
        t3 = time.perf_counter()
        
    elif op_name == "erode":
        k_size = kwargs.get("kernel_size", 3)
        kernel = np.ones((k_size, k_size), dtype=np.uint8)
        t0 = time.perf_counter()
        my_out = erode(gray, kernel_size=k_size, pad_mode="replicate")
        t1 = time.perf_counter()
        
        t2 = time.perf_counter()
        cv_out = cv2.erode(gray, kernel, borderType=cv2.BORDER_REPLICATE)
        t3 = time.perf_counter()
        
    elif op_name == "otsu":
        t0 = time.perf_counter()
        th_my = otsu_threshold(gray)
        my_out = (gray > th_my).astype(np.uint8) * 255
        t1 = time.perf_counter()
        
        t2 = time.perf_counter()
        th_cv, cv_out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t3 = time.perf_counter()
        result["note"] = f"Ngưỡng Otsu: NumPy={th_my} vs OpenCV={int(th_cv)}"
        
    else:
        raise ValueError(f"Không hỗ trợ thao tác đối chứng: {op_name}")
        
    result["numpy_time_ms"] = round((t1 - t0) * 1000.0, 2)
    result["opencv_time_ms"] = round((t3 - t2) * 1000.0, 2)
    result["numpy_output"] = my_out
    result["opencv_output"] = cv_out
    
    rms = calculate_rms_error(my_out, cv_out)
    result["e_rms"] = round(rms, 4)
    result["psnr_db"] = round(calculate_psnr(my_out, cv_out), 2)
    result["passed"] = (rms <= 1.0)
    
    if rms < 1e-4:
        result["status_msg"] = "✓ Trùng khớp chính xác 100% (e_rms = 0.0000)"
    elif rms <= 1.0:
        result["status_msg"] = f"✓ Trùng khớp chuẩn (sai lệch làm tròn <= 1 mức xám, e_rms = {rms:.4f})"
    else:
        result["status_msg"] = f"Sai số e_rms = {rms:.4f}"
        
    return result


def run_full_benchmark(image: np.ndarray = None) -> List[Dict[str, Any]]:
    """Chạy toàn bộ bộ nghiệm thu đối chứng với OpenCV."""
    if image is None:
        np.random.seed(42)
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        
    ops = ["median", "mean", "gaussian", "equalize_hist", "laplacian", "dilate", "erode", "otsu"]
    reports = []
    for op in ops:
        res = compare_operation(image, op)
        reports.append(res)
    return reports
