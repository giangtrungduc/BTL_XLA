# BÀI TẬP LỚN MÔN XỬ LÝ ẢNH (INT13146) — PTIT
## Đề tài: Xây dựng ứng dụng khôi phục và nâng cao chất lượng ảnh cũ bằng các kỹ thuật xử lý ảnh số

---

## 1. Giới thiệu

Ứng dụng desktop (PySide6) khôi phục ảnh cũ qua 4 bước cổ điển: xóa vết xước/bụi, khử nhiễu, chỉnh sáng – tương phản, làm nét.

- **Code from scratch (Chương 0):** mọi thuật toán xử lý ảnh trong `core/` tự cài đặt bằng NumPy, không import `cv2`. Pillow chỉ dùng để đọc/ghi tệp ảnh. OpenCV chỉ dùng trong `verify/compare_opencv.py` để đối chứng kết quả.
- **Giao diện:**
  - Ảnh gốc và kết quả đặt cạnh nhau, mỗi ảnh có histogram (mean, std, vạch phân vị 1%–99%).
  - Hộp **Hiển thị** ở ảnh kết quả cho xem ảnh **sau từng bước** và **mặt nạ vết xước** (tô đỏ).
  - Nút **Khôi phục tự động** đo đặc trưng ảnh rồi giải thích từng quyết định (vì sao bật/tắt bước nào).
  - 4 thẻ của quy trình: chọn phương pháp, chỉnh tham số, xem công thức ngay trên thẻ.
  - Bảng **PSNR / e_rms / SNR_ms trước → sau** khi có ảnh sạch để đối chiếu.
  - Nút **Tạo ảnh cũ mô phỏng**: lấy ảnh đang mở làm ảnh sạch, sinh bản bị bạc màu + nhiễu + vết xước để đo định lượng.
  - Ảnh lớn được thu nhỏ để xem trước mượt; khi **Lưu**, quy trình chạy lại trên ảnh gốc đủ độ phân giải.
  - Xử lý trong luồng nền (`QThread`), chỉ giữ yêu cầu mới nhất nên kéo thanh trượt không bị đơ.

---

## 2. Cài đặt và chạy

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

- Chạy ứng dụng: nhấp đúp `run.bat`, hoặc `.venv\Scripts\python.exe main.py`
- Chạy toàn bộ kiểm thử: `.venv\Scripts\python.exe tests\run_all_tests.py`
- In bảng đối chứng OpenCV: `.venv\Scripts\python.exe verify\compare_opencv.py`
- Sinh lại ảnh mẫu: `.venv\Scripts\python.exe data\create_samples.py`

---

## 3. Cấu trúc thư mục

```text
BTL_XLA/
├─ main.py, run.bat, requirements.txt
├─ core/                    # Thuật toán (chỉ NumPy)
│  ├─ padding.py            # Xử lý biên: mirror / replicate / zero (Ch.2)
│  ├─ window.py             # Cửa sổ trượt + tích chập "dịch rồi cộng" (Ch.2)
│  ├─ noise.py              # Mô hình nhiễu g = f + η: Gauss, đều, muối tiêu (Ch.2)
│  ├─ denoise.py            # 11 bộ lọc không gian (Ch.2)
│  ├─ intensity.py          # Gamma, log, âm bản, kéo giãn từng đoạn / tự động (Ch.2)
│  ├─ histogram.py          # Histogram, cân bằng histogram (Ch.2)
│  ├─ color.py              # RGB <-> HSI, xử lý trên kênh I (Ch.3)
│  ├─ sharpen.py            # Laplacian 4/8 lân cận, Sobel, làm nét có trọng số Sobel (Ch.2, Ch.6)
│  ├─ morphology.py         # Giãn, co, mở, đóng, gán nhãn thành phần liên thông (Ch.5)
│  ├─ scratch.py            # Otsu, phát hiện và điền vết xước (Ch.2, 5, 6)
│  ├─ metrics.py            # e_rms, SNR_ms, PSNR (Ch.4)
│  ├─ degrade.py            # Mô phỏng ảnh cũ từ ảnh sạch
│  ├─ resize.py             # Thu nhỏ bằng trung bình khối (cho khung xem trước)
│  └─ pipeline.py           # Stage, run_pipeline, analyze_image, diagnose_with_report
├─ ui/                      # Giao diện PySide6
│  ├─ theme.py              # Bảng màu, stylesheet
│  ├─ main_window.py        # Cửa sổ chính
│  ├─ image_pair.py         # Thẻ ảnh + histogram + hộp chọn chế độ xem
│  ├─ histogram_widget.py   # Vẽ histogram bằng QPainter
│  ├─ stage_card.py         # Thẻ điều khiển từng bước
│  └─ worker.py             # Luồng nền
├─ verify/compare_opencv.py # Đối chứng 9 phép toán với OpenCV
├─ tests/                   # test_core.py, test_ui.py, run_all_tests.py
├─ data/                    # clean/ (ảnh sạch), old/ (ảnh cũ thật và mô phỏng), create_samples.py
└─ Docs/                    # Slide bài giảng
```

---

## 4. Quy trình khôi phục (thứ tự cố định)

```
Ảnh gốc → 1. Vết xước → 2. Khử nhiễu → 3. Sáng & tương phản → 4. Làm nét → Kết quả
```

Mỗi lần chỉnh tham số, `run_pipeline()` chạy lại **từ ảnh gốc**, nên không bị cộng dồn kết quả cũ.

**Vì sao thứ tự này:**
1. **Vết xước trước khử nhiễu:** lọc trung vị làm vết xước mảnh bị mờ/đứt. Nếu phát hiện sau đó, mặt nạ dễ bắt nhầm vào biên thật của ảnh.
2. **Khử nhiễu trước tương phản:** kéo giãn histogram cũng kéo giãn biên độ nhiễu.
3. **Làm nét cuối cùng:** đạo hàm khuếch đại nhiễu (Ch.6), nên phải làm sạch trước.

| Bước | Phương pháp | Nguồn |
| :--- | :--- | :--- |
| **1. Vết xước** | Bản đồ đặc trưng = ảnh − trung vị 7×7 (mặc định chỉ lấy vệt **sáng**). Phân ngưỡng **Otsu**, ngưỡng có sàn theo mức nhiễu (MAD) để không bắt hạt nhiễu. **Đóng 3×3** nối vết đứt. **Thành phần liên thông 8 lân cận** giữ vệt dài–mảnh và đốm bụi nhỏ, bỏ mảng lớn đặc. Giãn 3×3. **Điền từ mép vào**: mỗi vòng điền các điểm ở mép mặt nạ bằng trung vị các điểm lành lân cận. | Ch.2 (lọc thống kê thứ tự), Ch.5 (hình thái học, liên thông), Ch.6 (Otsu) |
| **2. Khử nhiễu** | Trung vị, trung bình số học, Gauss, alpha-trimmed, k lân cận gần nhất, trung bình hình học, điều hòa, điều hòa ngược (Q), điểm giữa, max, min. Kernel 3/5/7. | Ch.2 |
| **3. Sáng & tương phản** | Kéo giãn tương phản (bỏ p% hai đầu histogram), cân bằng histogram, gamma, log, âm bản. Ảnh màu: chỉ xử lý kênh **I** của HSI để giữ màu. | Ch.2, Ch.3 |
| **4. Làm nét** | Laplacian 4 / 8 lân cận: g = f − k·∇²f. **Laplacian có trọng số Sobel**: g = f − k·W·∇²f, W = độ lớn gradient Sobel làm mịn 5×5, chuẩn hóa [0,1] → chỉ làm nét ở biên, không làm nổi hạt ở vùng phẳng. | Ch.2, Ch.6 |

---

## 5. Khôi phục tự động (`diagnose_with_report`)

`analyze_image()` đo các đặc trưng sau (trên ảnh xám):

| Đặc trưng | Cách đo |
| :--- | :--- |
| Tỉ lệ nhiễu xung | Điểm có giá trị ≤ 5 hoặc ≥ 250 **và** lệch > 50 so với trung vị 3×3 |
| σ nhiễu hạt | 1.4826 × MAD(ảnh − trung vị 3×3), bỏ các điểm nhiễu xung |
| Tỉ lệ vết xước | Tỉ lệ điểm thuộc các vệt dạng đường trong mặt nạ vết xước |
| Dải xám, độ sáng | Phân vị 1% và 99%, trung bình (đo trên ảnh đã lọc trung vị) |

Luật quyết định (hằng số ở đầu phần chẩn đoán trong `core/pipeline.py`):

| Điều kiện | Hành động |
| :--- | :--- |
| Vết xước > 0.2% | Bật bước 1 |
| Nhiễu xung > 0.3% và σ > 6 | Alpha-trimmed 3×3 (d = 4) |
| Chỉ nhiễu xung > 0.3% | Trung vị 3×3 (5×5 nếu > 12%) |
| Chỉ σ > 6 | Gauss 5×5, σ = 1 |
| Dải xám < 70 | Cân bằng histogram |
| Dải xám 70–180 | Kéo giãn tương phản (bỏ 1% hai đầu) |
| Dải đủ rộng, trung bình < 80 / > 175 | Gamma 0.75 / 1.3 |
| σ ≤ 6 | Làm nét có trọng số Sobel, k = 0.3 (ảnh còn nhiễu hạt thì không làm nét) |

---

## 6. Kết quả

### 6.1. Định lượng trên ảnh mô phỏng (có ảnh sạch)

Ảnh sạch `data/clean/sample_clean_portrait.png`, làm cũ bằng `simulate_old_photo`, khôi phục bằng nút tự động (số liệu in ra khi chạy `tests/test_core.py`):

| Kiểu hư hại | PSNR trước (dB) | PSNR sau (dB) |
| :--- | :---: | :---: |
| Nhiễu muối tiêu 3% | 17.30 | 30.63 |
| Nhiễu hạt Gauss (σ ≈ 15) | 24.53 | 30.76 |
| Bạc màu (co dải xám 50%) | 13.37 | 23.70 |
| 8 vết xước | 20.43 | 26.07 |
| Tổng hợp (bạc màu + nhiễu + vết xước) | 12.41 | 23.25 |

Phát hiện vết xước trên ảnh thử có đáp án: độ phủ (recall) ≈ 0.90–0.97, độ chính xác (precision) ≈ 0.8–0.9.

### 6.2. Đối chứng với OpenCV (`verify/compare_opencv.py`)

Chạy trên ảnh `vietnam_vintage_portrait.jpg` (ảnh xám):

| Phép toán | e_rms so với OpenCV |
| :--- | :---: |
| Trung vị 3×3 (`cv2.medianBlur`) | 0.0000 |
| Trung bình 3×3 (`cv2.blur`) | 0.0000 |
| Gauss 3×3, σ = 1 (`cv2.GaussianBlur`) | ≈ 0.09 (lệch làm tròn) |
| Cân bằng histogram (`cv2.equalizeHist`) | 0.0000 |
| Laplacian 8 lân cận (`cv2.filter2D`) | 0.0000 |
| Sobel \|Gx\| + \|Gy\| (`cv2.Sobel`) | 0.0000 |
| Giãn / co 3×3 (`cv2.dilate`, `cv2.erode`) | 0.0000 |
| Ngưỡng Otsu (`cv2.threshold`) | 0.0000 (cùng ngưỡng) |

### 6.3. Trường hợp thất bại / giới hạn

- **Mảng tróc lớn** (rộng hơn nhiều so với cửa sổ 7×7, ví dụ trên tóc và cổ áo của ảnh chân dung thật): không được phát hiện là vết xước vì luật hình dạng bỏ mảng lớn đặc để tránh xóa chi tiết sáng thật. Nếu tăng độ nhạy, vùng được điền sẽ bị nhòe.
- **Hai vết xước chạy sát nhau:** khi vết chiếm quá nửa cửa sổ trung vị, hiệu số gần 0 nên một phần vết bị bỏ sót.
- **Kéo giãn tương phản** không biết dải xám gốc của ảnh, nên PSNR sau khi khôi phục ảnh bạc màu chỉ khoảng 23–24 dB dù ảnh nhìn rõ hơn nhiều.
- **Làm nét** làm giảm nhẹ PSNR trên ảnh có biên cứng (gây vọng biên), đổi lại ảnh nhìn sắc hơn.
- **Ảnh màu:** HSI giữ sắc độ, nhưng khi tăng mạnh kênh I, một số điểm bị cắt ở 255 nên màu nhạt đi.

---

## 7. Ghi chú kỹ thuật

- Bộ lọc tuyến tính (trung bình, Gauss, Laplacian, Sobel) dùng `convolve2d` trong `core/window.py`: với mỗi hệ số w(s,t) của mặt nạ, dịch ảnh đã đệm biên đi (s,t), nhân w(s,t) rồi cộng dồn. Kết quả giống hệt nhân từng cửa sổ, nhưng chỉ tốn bộ nhớ O(H×W) nên chạy được ảnh vài chục megapixel.
- Bộ lọc thống kê thứ tự (trung vị, min, max, alpha-trimmed, …) dùng `sliding_window_view` của NumPy (chỉ là cách nhìn mảng, không sao chép dữ liệu).
- Ảnh ≥ 900 px được thu nhỏ bằng trung bình khối để xem trước; ảnh lưu ra luôn ở độ phân giải gốc (ảnh 12 MP mất khoảng 20–30 giây).
