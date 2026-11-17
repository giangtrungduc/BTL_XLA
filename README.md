# BÀI TẬP LỚN MÔN XỬ LÝ ẢNH (INT13146) — PTIT
## Đề tài: Xây dựng ứng dụng khôi phục và nâng cao chất lượng ảnh cũ bằng các kỹ thuật xử lý ảnh số

---

## 📌 1. Giới thiệu tổng quan
Dự án được xây dựng phục vụ bài tập lớn môn **Xử lý ảnh (INT13146)** — Học viện Công nghệ Bưu chính Viễn thông (PTIT).
- **Yêu cầu cốt lõi (Chương 0 tr.8):** 100% thuật toán xử lý ảnh được tự lập trình từ đầu bằng **NumPy (Code from scratch)**, tuyệt đối không dùng các hàm xử lý có sẵn của OpenCV trong package `core/`. OpenCV chỉ được sử dụng ở module `verify/compare_opencv.py` để đối chứng nghiệm thu và chứng minh tính đúng đắn của thuật toán.
- **Giao diện Desktop (PySide6):** Thiết kế tối giản, thanh lịch bám sát bài giảng và ảnh mock-up quy chuẩn:
  - Cột trái: Hiển thị song song **1. Ảnh gốc** (+ Histogram) và **2. Kết quả** (+ Histogram & Badge trạng thái).
  - Cột phải: **Quy trình khôi phục có thứ tự cố định** gồm 4 thẻ độc lập: Khử nhiễu $\rightarrow$ Vết xước $\rightarrow$ Sáng và tương phản $\rightarrow$ Làm nét.
  - Nút lớn **⚡ Khôi phục tự động**: Chẩn đoán tự động tổn hại của ảnh và kích hoạt quy trình tối ưu.
  - Bảng chỉ số định lượng: Sai số căn quân phương ($e_{rms}$) và Tỷ số tín hiệu/nhiễu ($SNR_{ms}$).
  - Chạy ngầm bằng `QThread` (`worker.py`) giúp giao diện luôn mượt mà, không bị đơ giật khi kéo thanh trượt.

---

## 📂 2. Cấu trúc thư mục dự án

```text
restore_app/ (d:\Nam4\XLA\BTL_XLA\)
├─ main.py                  # Điểm khởi chạy ứng dụng Desktop chính
├─ run.bat                  # Phím tắt click đúp 1 chạm để chạy app trên Windows
├─ requirements.txt         # Danh mục thư viện phụ thuộc
├─ core/                    # Toàn bộ thuật toán CODE FROM SCRATCH bằng NumPy (Không import cv2)
│  ├─ __init__.py           # Export toàn bộ API của package core
│  ├─ padding.py            # Xử lý biên: mirror / replicate / zero (Ch.2 tr.91-96)
│  ├─ window.py             # Tiện ích sliding_window_view dùng chung (vectorized)
│  ├─ noise.py              # Mô hình nhiễu: g = f + η (Gauss, đều, muối tiêu - Ch.2 tr.128-132)
│  ├─ denoise.py            # Bộ lọc không gian: median, mean số học, hình học, điều hòa,
│  │                        # điều hòa ngược (Q), max/min, midpoint, alpha-trimmed,
│  │                        # k-nearest mean, Gaussian lowpass (Ch.2 tr.91-117, 133-141)
│  ├─ intensity.py          # Biến đổi điểm ảnh: gamma, log, âm bản, kéo giãn từng đoạn (Ch.2 tr.43-56)
│  ├─ histogram.py          # Histogram, cân bằng CDF (cả 2 phương pháp - Ch.2 tr.58-78)
│  ├─ color.py              # Chuyển đổi RGB <-> HSI, xử lý kênh I bảo toàn màu (Ch.3 tr.32-33, 53)
│  ├─ sharpen.py            # Làm nét: Laplacian 4 & 8 lân cận, Sobel, chế độ kết hợp (Ch.2 & Ch.6)
│  ├─ morphology.py         # Hình thái học: dilate, erode, open, close, connected components (Ch.5)
│  ├─ scratch.py            # Giảm vết xước: otsu, detect_mask, fill_masked (Ch.6, Ch.5, Ch.2 tr.116)
│  ├─ metrics.py            # Đo lường định lượng: e_rms, SNR_ms, PSNR (Ch.4 tr.68-70)
│  ├─ degrade.py            # Mô phỏng ảnh cũ từ ảnh sạch (thêm nhiễu, bạc màu, vết xước)
│  └─ pipeline.py           # Stage(enabled, method, params), run_pipeline, diagnose
├─ verify/
│  └─ compare_opencv.py     # Nghiệm thu đối chứng 8 thuật toán với OpenCV C++ (Ch.0 tr.10)
├─ ui/                      # Giao diện đồ họa Desktop (PySide6)
│  ├─ __init__.py
│  ├─ main_window.py        # Cửa sổ chính chia cột chuẩn mock-up
│  ├─ image_pair.py         # Cặp thẻ hiển thị ảnh Before/After và biểu đồ Histogram
│  ├─ histogram_widget.py   # Widget vẽ biểu đồ Histogram trực quan bằng QPainter
│  ├─ stage_card.py         # Thẻ card tương tác cho từng bước khôi phục
│  └─ worker.py             # QThread xử lý nền, không làm đơ giao diện
├─ tests/                   # Bộ kiểm thử tự động
│  ├─ __init__.py
│  ├─ test_core.py          # Unit test toàn bộ các module trong core/
│  ├─ test_ui.py            # Kiểm thử khởi tạo và tương tác GUI headless
│  └─ run_all_tests.py      # Script chạy toàn bộ test tự động (All-in-one)
├─ data/                    # Dữ liệu ảnh phục vụ thử nghiệm
│  ├─ clean/                # Ảnh sạch ban đầu
│  ├─ old/                  # Ảnh cũ thực tế (bao gồm ảnh chân dung lịch sử) và ảnh giả lập
│  └─ create_samples.py     # Script tự động sinh ảnh mẫu
└─ Docs/                    # Toàn bộ slide và giáo trình bài giảng môn học
```

---

## 🧮 3. Bảng chức năng, thuật toán & nguồn trích dẫn giáo trình

| Chức năng | Thuật toán & Cơ sở lý thuyết | Nguồn tài liệu |
| :--- | :--- | :--- |
| **Ảnh màu** | Chuyển RGB sang HSI, chỉ xử lý trên kênh $I$ (Intensity), giữ nguyên $H$ và $S$. Cân bằng riêng từng kênh RGB sẽ làm sai lệch sắc độ nghiêm trọng. | **Chương 3** tr.32–33, 53 |
| **Chẩn đoán ảnh** | Phân tích Histogram, mean, std. Tỷ lệ điểm ảnh bằng $0$ hoặc $255$ cao gợi ý nhiễu xung muối tiêu. Độ lệch chuẩn hẹp gợi ý ảnh thiếu tương phản. | **Chương 2** tr.58–62, 129 |
| **Khử nhiễu** | 11 bộ lọc không gian: Median, trung bình số học, trung bình hình học, điều hòa, điều hòa ngược (tham số $Q$), cực đại / cực tiểu, midpoint, alpha-trimmed mean, trung bình $k$ lân cận gần nhất, Gaussian lowpass. Ba chế độ xử lý biên: *mirror*, *replicate*, *zero padding*. | **Chương 2** tr.91–117, 133–141 |
| **Sáng & Tương phản** | Biến đổi Gamma ($s = c \cdot r^\gamma$), Biến đổi Logarit ($s = c \cdot \log(1+r)$), Âm bản ($255 - r$), Kéo giãn từng đoạn (Piecewise linear), Cân bằng Histogram (hỗ trợ cả phương pháp giáo trình slide 70 và phương pháp kéo đáy về 0 khớp OpenCV). | **Chương 2** tr.43–56, 65–78 |
| **Làm nét** | Đạo hàm bậc 2 Laplacian 4 và 8 lân cận ($f - k \cdot \nabla^2 f$), Độ lớn Gradient Sobel ($|G_x| + |G_y|$), Chế độ kết hợp nâng cao kinh điển: Laplacian + Sobel + Smooth $5 \times 5$ + Nhân mặt nạ + Cộng ảnh gốc + Gamma. | **Chương 2** tr.119–123;<br>**Chương 6** tr.20–28, 58–60 |
| **Giảm vết xước** | Phát hiện vết: lấy ảnh trừ ảnh đã lọc Median hoặc dùng $|\text{Laplacian}|$, phân ngưỡng Otsu cực đại hóa phương sai giữa các lớp. Làm sạch mặt nạ bằng phép đóng, giãn, mở và thành phần liên thông 8 lân cận. Điền vết bằng Median hoặc $k$-lân cận gần nhất, **chỉ lấy các điểm lành lặn ngoài mặt nạ**. | **Chương 6** tr.64–65, 71, 81+;<br>**Chương 5** tr.23–70;<br>**Chương 2** tr.116 |
| **Đánh giá** | Sai số căn quân phương $e_{rms}$, Tỷ số tín hiệu/nhiễu $SNR_{ms}$, PSNR, nghiệm thu đối chứng OpenCV C++. | **Chương 4** tr.68–70;<br>**Chương 0** tr.10 |
| **Mô phỏng ảnh cũ** | Mô hình suy thoái $g = f + \eta$ (Gaussian, Uniform, Muối tiêu), giảm tương phản bằng Gamma, vẽ vết xước và bụi phim ngẫu nhiên. | **Chương 2** tr.128–132;<br>**Chương 0** tr.8 |

> **Lưu ý học thuật về phần "Điền vết xước":**  
> Đây là phương pháp được ghép từ các bộ lọc thống kê lân cận ngoài mặt nạ thuộc Chương 2, dùng thay thế cho kỹ thuật Inpainting chuyên sâu. Thuật toán xử lý rất xuất sắc các vết xước sợi mảnh và nứt rạn chân chim; chưa xử lý được các vết rách thủng hay vết ố quá lớn (đây là giới hạn tự nhiên của các kỹ thuật xử lý ảnh truyền thống).

---

## ⚡ 4. Nguyên tắc thiết kế Pipeline & Khôi phục tự động

1. **Hàm thuần khiết ($f(\text{ảnh}, \text{tham số}) \rightarrow \text{ảnh mới}$):**  
   Mỗi lần người dùng điều chỉnh thanh trượt hoặc bật/tắt bước, hàm `run_pipeline()` luôn **chạy lại từ ảnh gốc ban đầu theo thứ tự 4 bước cố định**:
   $$\text{Ảnh gốc} \xrightarrow{\text{1. Khử nhiễu}} \text{Ảnh mịn} \xrightarrow{\text{2. Vết xước}} \text{Ảnh sạch vết} \xrightarrow{\text{3. Tương phản}} \text{Ảnh rõ nét} \xrightarrow{\text{4. Làm nét}} \text{Kết quả}$$
   - **Lý do khoa học (Chương 6 tr.29–31):** Đạo hàm làm nét khuếch đại nhiễu cực mạnh, do đó bắt buộc phải làm mịn và xóa vết nứt trước khi làm nét.
   - **Loại bỏ hoàn toàn lỗi "dồn bước":** Kéo thanh trượt không bị cộng dồn chồng chéo kết quả cũ, người dùng tự do tinh chỉnh mà không cần bấm Hoàn tác/Undo.
2. **Nút "⚡ Khôi phục tự động" (`diagnose()`):**  
   Tự động chẩn đoán tổn hại ảnh dựa trên các luật thực nghiệm:
   - Tỷ lệ điểm cực trị ($0$ hoặc $255$) $> 0.8\%$ $\rightarrow$ Bật Lọc trung vị $3 \times 3$ để triệt tiêu nhiễu muối tiêu.
   - Phát hiện vết xước/nứt $> 0.5\%$ $\rightarrow$ Bật Giảm vết xước (điền nội suy ngoài mặt nạ).
   - Độ lệch chuẩn $\text{std} < 38$ (histogram dồn cục bộ) $\rightarrow$ Bật Cân bằng Histogram.
   - Mean $< 95$ (ảnh bị tối) $\rightarrow$ Bật Gamma $\gamma = 0.80$ để tăng sáng vùng bóng râm.
   - Mean $> 165$ (ảnh bị bạc màu, chói) $\rightarrow$ Bật Gamma $\gamma = 1.25$ để làm đậm nét.
   - Bật Làm nét viền cạnh nhẹ nhàng (Laplacian 4 lân cận, $k = 0.30$).

---

## 🚀 5. Hướng dẫn khởi chạy ứng dụng

### Cách 1: Click đúp 1 chạm (Khuyên dùng)
Nhấp đúp chuột vào file **`run.bat`** ở thư mục gốc. Ứng dụng sẽ tự động kích hoạt môi trường `.venv` và khởi động giao diện Desktop.

### Cách 2: Chạy từ Terminal / PowerShell
```powershell
& .venv\Scripts\python.exe main.py
```

### Chạy toàn bộ kiểm thử đơn vị & đối chứng OpenCV
```powershell
& .venv\Scripts\python.exe tests/run_all_tests.py
```

---

## 📊 6. Kết quả đối chứng nghiệm thu với OpenCV C++ (Chương 0 tr.10)

| Thao tác kiểm thử | Thời gian NumPy (ms) | Thời gian OpenCV (ms) | Sai số $e_{rms}$ | Đánh giá nghiệm thu |
| :--- | :---: | :---: | :---: | :--- |
| **Lọc Trung Vị (Median 3x3)** | $6.95\text{ ms}$ | $0.15\text{ ms}$ | **$0.0000$** | **Khớp chính xác 100% với cv2.medianBlur** |
| **Lọc Trung Bình (Mean 3x3)** | $1.88\text{ ms}$ | $1.76\text{ ms}$ | **$0.0000$** | **Khớp chính xác 100% với cv2.blur** |
| **Lọc Gaussian ($\sigma = 1.0$)** | $4.33\text{ ms}$ | $0.74\text{ ms}$ | **$0.2901$** | **Khớp chuẩn (lệch làm tròn $\le 1$ mức xám)** |
| **Cân bằng Histogram (CDF)** | $0.81\text{ ms}$ | $0.08\text{ ms}$ | **$0.0000$** | **Khớp chính xác 100% với cv2.equalizeHist** |
| **Làm nét Laplacian (8 lân cận)** | $4.24\text{ ms}$ | $0.45\text{ ms}$ | **$0.0000$** | **Khớp chính xác 100% với cv2.filter2D** |
| **Phép giãn (Dilation 3x3)** | $4.68\text{ ms}$ | $0.13\text{ ms}$ | **$0.0000$** | **Khớp chính xác 100% với cv2.dilate** |
| **Phép co (Erosion 3x3)** | $3.41\text{ ms}$ | $0.07\text{ ms}$ | **$0.0000$** | **Khớp chính xác 100% với cv2.erode** |
| **Phân ngưỡng Otsu** | $0.33\text{ ms}$ | $0.07\text{ ms}$ | **$0.0000$** | **Khớp chính xác 100% với cv2.threshold (OTSU)** |
