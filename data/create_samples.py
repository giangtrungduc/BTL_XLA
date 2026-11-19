"""
Script data/create_samples.py
Tạo ảnh mẫu phục vụ thử nghiệm và demo (BTL Xử lý ảnh INT13146 - PTIT).

Chạy từ thư mục gốc dự án:  python data/create_samples.py
Sinh ra:
  data/clean/sample_clean_portrait.png   ảnh sạch (ground truth), dải xám rộng 15..235
  data/old/sample_simulated_old.png      ảnh cũ mô phỏng: bạc màu + nhiễu + vết xước
  data/old/sample_low_contrast.png       ảnh tương phản rất thấp (dải 40..95)
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from PIL import Image

from core.degrade import simulate_old_photo


def make_clean_portrait(h: int = 360, w: int = 360) -> np.ndarray:
    """Vẽ một chân dung đơn giản có chuyển sắc (dùng làm ảnh sạch có đáp án)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 70 + 70 * (yy / h)                                   # nền chuyển sắc 70..140
    cy, cx = h / 2 - 25, w / 2
    face = ((xx - cx) / 68) ** 2 + ((yy - cy) / 82) ** 2 <= 1.0
    shade = 215 - 45 * np.sqrt(((xx - cx + 25) / 68) ** 2 + ((yy - cy + 25) / 82) ** 2)
    img[face] = np.clip(shade[face], 150, 235)                 # khuôn mặt có đổ bóng
    hair = (((xx - cx) / 78) ** 2 + ((yy - cy + 12) / 92) ** 2 <= 1.0) & ~face & (yy < cy + 10)
    img[hair] = 30                                             # tóc
    for ex in (cx - 30, cx + 30):                              # mắt
        eye = ((xx - ex) / 13) ** 2 + ((yy - (cy - 12)) / 7) ** 2 <= 1.0
        img[eye] = 20
    nose = (np.abs(xx - cx) <= 1) & (yy > cy - 5) & (yy < cy + 18)
    img[nose] = 120
    mouth = (np.abs(((xx - cx) / 26) ** 2 + ((yy - (cy + 28)) / 10) ** 2 - 1.0) < 0.18) & (yy > cy + 28)
    img[mouth] = 90
    body = (yy > h / 2 + 55) & (np.abs(xx - cx) < (yy - (h / 2 + 55)) * 1.1 + 40)
    img[body] = 25                                             # áo
    collar = body & (np.abs(xx - cx) < 14) & (yy < h / 2 + 110)
    img[collar] = 230
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def create_all_samples():
    clean_dir = os.path.join(ROOT_DIR, "data", "clean")
    old_dir = os.path.join(ROOT_DIR, "data", "old")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(old_dir, exist_ok=True)

    clean_arr = make_clean_portrait()
    clean_path = os.path.join(clean_dir, "sample_clean_portrait.png")
    Image.fromarray(clean_arr).save(clean_path)
    print(f"Đã tạo: {clean_path}")

    old_sim, _ = simulate_old_photo(
        clean_arr, noise_type="both", noise_level=0.015, gamma_val=1.0,
        num_scratches=6, seed=100, fade=0.45,
    )
    old_sim_path = os.path.join(old_dir, "sample_simulated_old.png")
    Image.fromarray(old_sim).save(old_sim_path)
    print(f"Đã tạo: {old_sim_path}")

    rng = np.random.default_rng(7)
    low_arr = np.clip(rng.normal(68, 10, (300, 300)), 40, 95).astype(np.uint8)
    low_arr[100:200, 100:200] = 85
    low_arr[130:170, 130:170] = 52
    low_path = os.path.join(old_dir, "sample_low_contrast.png")
    Image.fromarray(low_arr).save(low_path)
    print(f"Đã tạo: {low_path}")


if __name__ == "__main__":
    create_all_samples()
