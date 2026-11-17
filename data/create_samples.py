"""
Script data/create_samples.py
Tạo các ảnh mẫu phục vụ thử nghiệm và demo BTL Xử lý ảnh (INT13146 - PTIT).
"""
import os
import sys

# Đảm bảo import được package core
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from PIL import Image, ImageDraw

from core.degrade import simulate_old_photo


def create_all_samples():
    clean_dir = os.path.join("data", "clean")
    old_dir = os.path.join("data", "old")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(old_dir, exist_ok=True)

    # 1. Tạo ảnh chân dung sạch mẫu (Clean Portrait)
    h, w = 360, 360
    img = Image.new("L", (w, h), color=80)
    draw = ImageDraw.Draw(img)

    # Nền gradient
    for y in range(h):
        val = int(70 + 50 * (y / h))
        draw.line([(0, y), (w, y)], fill=val)

    # Vẽ khuôn mặt và mắt mũi miệng
    draw.ellipse([w//2 - 65, h//2 - 95, w//2 + 65, h//2 + 45], fill=160, outline=120)
    draw.ellipse([w//2 - 45, h//2 - 50, w//2 - 20, h//2 - 35], fill=40)  # Mắt trái
    draw.ellipse([w//2 + 20, h//2 - 50, w//2 + 45, h//2 - 35], fill=40)  # Mắt phải
    draw.line([w//2, h//2 - 30, w//2, h//2 - 8], fill=90, width=2)       # Mũi
    draw.arc([w//2 - 25, h//2 - 5, w//2 + 25, h//2 + 15], start=0, end=180, fill=60, width=2) # Miệng
    draw.polygon([(w//2 - 110, h), (w//2, h//2 + 45), (w//2 + 110, h)], fill=45) # Áo vest

    clean_arr = np.array(img, dtype=np.uint8)
    clean_path = os.path.join(clean_dir, "sample_clean_portrait.png")
    Image.fromarray(clean_arr).save(clean_path)
    print(f"Đã tạo: {clean_path}")

    # 2. Tạo ảnh cũ giả lập từ ảnh sạch (có nhiễu muối tiêu + vết xước + giảm tương phản)
    old_sim, _ = simulate_old_photo(
        clean_arr,
        noise_type="both",
        noise_level=0.015,
        gamma_val=1.25,
        num_scratches=6,
        seed=100
    )
    old_sim_path = os.path.join(old_dir, "sample_simulated_old.png")
    Image.fromarray(old_sim).save(old_sim_path)
    print(f"Đã tạo: {old_sim_path}")

    # 3. Tạo ảnh tương phản cực thấp (Low contrast: dải hẹp [40, 95])
    low_arr = np.clip(np.random.normal(68, 10, (300, 300)), 40, 95).astype(np.uint8)
    low_arr[100:200, 100:200] = 85
    low_arr[130:170, 130:170] = 52
    low_path = os.path.join(old_dir, "sample_low_contrast.png")
    Image.fromarray(low_arr).save(low_path)
    print(f"Đã tạo: {low_path}")


if __name__ == "__main__":
    create_all_samples()
