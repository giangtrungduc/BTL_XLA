"""
Unit test kiểm thử toàn diện các module của core/
"""
import os
import sys

# Đảm bảo nạp được thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import core


def test_padding():
    img = np.ones((10, 10), dtype=np.uint8)
    p_mirror = core.pad_image(img, 2, 2, mode="mirror")
    assert p_mirror.shape == (14, 14)
    p_rep = core.pad_image(img, 2, 2, mode="replicate")
    assert p_rep.shape == (14, 14)
    p_zero = core.pad_image(img, 2, 2, mode="zero")
    assert p_zero.shape == (14, 14)
    assert p_zero[0, 0] == 0
    print("✓ test_padding: PASSED")


def test_filters():
    np.random.seed(42)
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    
    # 1. Median
    m = core.median_filter(img, 3)
    assert m.shape == (32, 32)
    
    # 2. Arithmetic mean
    am = core.arithmetic_mean_filter(img, 3)
    assert am.shape == (32, 32)
    
    # 3. Geometric mean
    gm = core.geometric_mean_filter(img, 3)
    assert gm.shape == (32, 32)
    
    # 4. Harmonic mean
    hm = core.harmonic_mean_filter(img, 3)
    assert hm.shape == (32, 32)
    
    # 5. Contraharmonic mean
    chm = core.contraharmonic_mean_filter(img, 3, Q=1.5)
    assert chm.shape == (32, 32)
    
    # 6. Min/Max/Midpoint
    mn = core.min_filter(img, 3)
    mx = core.max_filter(img, 3)
    mp = core.midpoint_filter(img, 3)
    assert np.all(mn <= mx)
    
    # 7. Alpha-trimmed mean
    atm = core.alpha_trimmed_mean_filter(img, 3, d=2)
    assert atm.shape == (32, 32)
    
    # 8. K-nearest mean
    knm = core.k_nearest_mean_filter(img, 3, k=5)
    assert knm.shape == (32, 32)
    
    # 9. Gaussian
    gf = core.gaussian_filter(img, 3, sigma=1.0)
    assert gf.shape == (32, 32)
    
    print("✓ test_filters: PASSED")


def test_intensity_and_histogram():
    img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
    
    # Gamma
    g = core.gamma_transform(img, gamma=0.5)
    assert g[0, 0] > img[0, 0]
    
    # Log
    lg = core.log_transform(img)
    assert lg.shape == (2, 2)
    
    # Negative
    neg = core.negative_transform(img)
    assert np.all(neg == 255 - img)
    
    # Histogram
    h, p = core.compute_histogram(img)
    assert len(h) == 256
    assert np.isclose(np.sum(p), 1.0)
    
    # Equalize (cả 2 phương pháp)
    eq1 = core.equalize_histogram(img, method="stretch_to_zero")
    eq2 = core.equalize_histogram(img, method="textbook")
    assert eq1.shape == (2, 2)
    assert eq2.shape == (2, 2)
    
    print("✓ test_intensity_and_histogram: PASSED")


def test_color_hsi():
    rgb = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
    h, s, i = core.rgb_to_hsi(rgb)
    assert h.shape == (20, 20)
    assert s.shape == (20, 20)
    assert i.shape == (20, 20)
    
    # Chuyển ngược lại RGB
    rgb_reconstructed = core.hsi_to_rgb(h, s, i)
    diff = np.abs(rgb.astype(int) - rgb_reconstructed.astype(int))
    # Sai lệch chuyển đổi màu do làm tròn số học <= 2 mức xám
    assert np.max(diff) <= 2, f"Sai lệch HSI <-> RGB quá lớn: {np.max(diff)}"
    
    # Test xử lý trên kênh I
    def _eq_proc(gray):
        return core.equalize_histogram(gray)
    res_color = core.process_color_on_intensity(rgb, _eq_proc)
    assert res_color.shape == (20, 20, 3)
    
    print("✓ test_color_hsi: PASSED")


def test_sharpening():
    img = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
    
    # Laplacian 4 & 8
    lap4 = core.laplacian_sharpen(img, k=0.4, mode="4-neighbor")
    lap8 = core.laplacian_sharpen(img, k=0.4, mode="8-neighbor")
    assert lap4.shape == (30, 30)
    assert lap8.shape == (30, 30)
    
    # Sobel
    sob = core.sobel_filter(img)
    assert sob.shape == (30, 30)
    
    # Combined mode
    comb = core.combined_sharpen(img, k_lap=0.4, gamma_val=0.85)
    assert comb.shape == (30, 30)
    
    print("✓ test_sharpening: PASSED")


def test_morphology_and_scratch():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 10] = 255  # 1 vạch dọc 10 pixel
    
    dil = core.dilate(mask, 3)
    assert np.sum(dil > 0) > np.sum(mask > 0)
    
    ero = core.erode(dil, 3)
    assert np.sum(ero > 0) <= np.sum(dil > 0)
    
    filtered, num = core.connected_components(mask, min_size=5)
    assert num == 1
    
    # Test fill_masked
    img = np.full((20, 20), 100, dtype=np.uint8)
    img[mask > 0] = 255 # vạch xước trắng
    filled = core.fill_masked(img, mask, method="median", kernel_size=5)
    # Các điểm xước phải được lấp bằng 100 (giá trị nền lành lặn)
    assert np.all(filled[mask > 0] == 100)
    
    print("✓ test_morphology_and_scratch: PASSED")


def test_pipeline_and_diagnose():
    img = np.random.randint(50, 150, (60, 60), dtype=np.uint8)
    stages = core.diagnose(img)
    assert len(stages) == 4
    
    out = core.run_pipeline(img, stages)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    
    print("✓ test_pipeline_and_diagnose: PASSED")


if __name__ == "__main__":
    print("==================================================")
    print("   CHẠY UNIT TEST TOÀN BỘ CÁC MODULE TRONG CORE   ")
    print("==================================================")
    test_padding()
    test_filters()
    test_intensity_and_histogram()
    test_color_hsi()
    test_sharpening()
    test_morphology_and_scratch()
    test_pipeline_and_diagnose()
    print("==================================================")
    print("   ✓✓✓ 100% TẤT CẢ UNIT TEST TRONG CORE ĐÃ PASS!  ")
    print("==================================================")
