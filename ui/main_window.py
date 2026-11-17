"""
Module ui/main_window.py
Cửa sổ chính của ứng dụng Khôi phục và Nâng cao chất lượng ảnh cổ (restore_app).
Thiết kế chuẩn xác 100% theo giao diện người dùng yêu cầu:
  - Header: [ Mở ảnh ] [ Ảnh mẫu ] ... [ Lưu kết quả ]
  - Khung trái: 1. Ảnh gốc (+ Histogram) | 2. Kết quả (+ Histogram & Badge)
  - Khung phải: Quy trình khôi phục thứ tự cố định (4 thẻ: Khử nhiễu, Vết xước, Sáng & Tương phản, Làm nét)
  - Nút lớn nổi bật: [ ⚡ Khôi phục tự động ]
  - Khung chỉ số định lượng: e_rms và SNR_ms (tính khi có ảnh sạch đối chiếu)
  - Xử lý nền bằng QThread không làm đơ giao diện khi kéo thanh trượt.
"""
import os
import time
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QFrame
)
from PySide6.QtGui import QFont, QIcon, QColor
from PySide6.QtCore import Qt

from core.pipeline import Stage, diagnose
from core.metrics import calculate_rms_error, calculate_snr_ms
from ui.image_pair import ImageCard
from ui.stage_card import StageCard
from ui.worker import PipelineWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Khôi phục & Nâng cao chất lượng ảnh cũ — BTL Xử lý ảnh (INT13146 - PTIT)")
        self.resize(1180, 760)
        self.setMinimumSize(980, 680)

        # Trạng thái dữ liệu ảnh (mảng numpy uint8)
        self.orig_image: np.ndarray = None
        self.processed_image: np.ndarray = None
        self.clean_ground_truth: np.ndarray = None  # Ảnh sạch dùng đối chiếu nếu có

        # Khởi tạo Worker luồng nền
        self.worker = PipelineWorker(self)
        self.worker.result_ready.connect(self._on_worker_result)
        self.worker.error_occurred.connect(self._on_worker_error)

        # Áp dụng giao diện sáng hiện đại như trong mock-up
        self._apply_theme()

        # Dựng cây giao diện
        self._setup_ui()

        # Tự động nạp ảnh mẫu chân dung thực tế khi khởi động
        default_sample = os.path.join("data", "old", "vietnam_vintage_portrait.jpg")
        if not os.path.exists(default_sample):
            default_sample = os.path.join("data", "old", "sample_simulated_old.png")
            
        if os.path.exists(default_sample):
            self.load_image_from_path(default_sample)

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8fafc;
            }
            QWidget {
                font-family: 'Segoe UI', -apple-system, sans-serif;
                color: #1e293b;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 500;
                font-size: 12px;
                color: #334155;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
                border-color: #94a3b8;
            }
            QPushButton#btn-auto {
                background-color: #090d16;
                color: #ffffff;
                font-weight: bold;
                font-size: 13px;
                border-radius: 8px;
                padding: 10px 16px;
                border: 1px solid #1e293b;
            }
            QPushButton#btn-auto:hover {
                background-color: #1e293b;
            }
            QPushButton#btn-auto:pressed {
                background-color: #0284c7;
            }
            QPushButton#btn-save {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                font-weight: 600;
                color: #0f172a;
            }
            QPushButton#btn-save:hover {
                background-color: #f8fafc;
                border-color: #0284c7;
                color: #0284c7;
            }
        """)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 16, 20, 14)
        main_layout.setSpacing(12)

        # ========================================================
        # 1. HEADER TOOLBAR
        # ========================================================
        header = QHBoxLayout()
        header.setSpacing(10)

        self.btn_open = QPushButton("📁 Mở ảnh")
        self.btn_open.clicked.connect(self._on_open_file)
        header.addWidget(self.btn_open)

        self.btn_sample = QPushButton("🖼️ Ảnh mẫu")
        self.btn_sample.clicked.connect(self._on_load_sample)
        header.addWidget(self.btn_sample)

        header.addStretch()

        self.btn_save = QPushButton("💾 Lưu kết quả")
        self.btn_save.setObjectName("btn-save")
        self.btn_save.clicked.connect(self._on_save_file)
        header.addWidget(self.btn_save)

        main_layout.addLayout(header)

        # ========================================================
        # 2. BODY CHÍNH (Chia 2 khối: Khối Ảnh bên trái - Khối Quy trình bên phải)
        # ========================================================
        body_layout = QHBoxLayout()
        body_layout.setSpacing(14)

        # --- KHỐI TRÁI: 2 Ảnh song song (Ảnh gốc & Ảnh kết quả) ---
        images_container = QHBoxLayout()
        images_container.setSpacing(12)

        self.card_orig = ImageCard("1. Ảnh gốc", is_result=False)
        images_container.addWidget(self.card_orig, 1)

        self.card_proc = ImageCard("2. Kết quả", is_result=True)
        images_container.addWidget(self.card_proc, 1)

        body_layout.addLayout(images_container, 65)

        # --- KHỐI PHẢI: Quy trình khôi phục cố định (4 thẻ + Nút Tự động + Metrics) ---
        pipeline_panel = QVBoxLayout()
        pipeline_panel.setSpacing(8)

        # Tiêu đề cột phải
        lbl_pipeline_title = QLabel("Quy trình khôi phục (thứ tự cố định)")
        lbl_pipeline_title.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")
        pipeline_panel.addWidget(lbl_pipeline_title)

        # 4 Thẻ quy trình
        self.card_denoise = StageCard("denoise", "1. Khử nhiễu")
        self.card_denoise.changed.connect(self._trigger_pipeline)
        pipeline_panel.addWidget(self.card_denoise)

        self.card_scratch = StageCard("scratch", "2. Vết xước")
        self.card_scratch.changed.connect(self._trigger_pipeline)
        pipeline_panel.addWidget(self.card_scratch)

        self.card_contrast = StageCard("contrast", "3. Sáng và tương phản")
        self.card_contrast.changed.connect(self._trigger_pipeline)
        pipeline_panel.addWidget(self.card_contrast)

        self.card_sharpen = StageCard("sharpen", "4. Làm nét")
        self.card_sharpen.changed.connect(self._trigger_pipeline)
        pipeline_panel.addWidget(self.card_sharpen)

        # Nút lớn: Khôi phục tự động
        self.btn_auto = QPushButton("⚡ Khôi phục tự động")
        self.btn_auto.setObjectName("btn-auto")
        self.btn_auto.setCursor(Qt.PointingHandCursor)
        self.btn_auto.clicked.connect(self._on_auto_restore)
        pipeline_panel.addWidget(self.btn_auto)

        # Hộp hiển thị chỉ số đo lường e_rms và SNR_ms
        metrics_box = QFrame()
        metrics_box.setStyleSheet("background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 4px;")
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(10, 6, 10, 6)
        metrics_layout.setSpacing(4)

        row_metrics = QHBoxLayout()
        
        box_rms = QVBoxLayout()
        lbl_rms_title = QLabel("e_rms")
        lbl_rms_title.setStyleSheet("font-size: 10px; color: #64748b; font-weight: 500;")
        self.lbl_rms_val = QLabel("--")
        self.lbl_rms_val.setStyleSheet("font-size: 13px; font-weight: bold; color: #0284c7; font-family: Consolas;")
        box_rms.addWidget(lbl_rms_title)
        box_rms.addWidget(self.lbl_rms_val)
        row_metrics.addLayout(box_rms)

        box_snr = QVBoxLayout()
        lbl_snr_title = QLabel("SNR_ms")
        lbl_snr_title.setStyleSheet("font-size: 10px; color: #64748b; font-weight: 500;")
        self.lbl_snr_val = QLabel("--")
        self.lbl_snr_val.setStyleSheet("font-size: 13px; font-weight: bold; color: #059669; font-family: Consolas;")
        box_snr.addWidget(lbl_snr_title)
        box_snr.addWidget(self.lbl_snr_val)
        row_metrics.addLayout(box_snr)

        metrics_layout.addLayout(row_metrics)
        pipeline_panel.addWidget(metrics_box)

        lbl_metric_note = QLabel("Chỉ số hiện khi có ảnh sạch để đối chiếu.")
        lbl_metric_note.setStyleSheet("font-size: 10px; color: #94a3b8; font-style: italic;")
        pipeline_panel.addWidget(lbl_metric_note)

        body_layout.addLayout(pipeline_panel, 35)
        main_layout.addLayout(body_layout)

        # ========================================================
        # 3. FOOTER
        # ========================================================
        lbl_footer = QLabel("Mỗi lần chỉnh, ảnh được tính lại từ ảnh gốc. Khung xem trước chạy trên bản thu nhỏ.")
        lbl_footer.setStyleSheet("font-size: 11px; color: #64748b; padding-top: 2px;")
        main_layout.addWidget(lbl_footer)

    # --------------------------------------------------------
    # QUẢN LÝ DỮ LIỆU VÀ NẠP ẢNH
    # --------------------------------------------------------
    def load_image_from_path(self, path: str, clean_reference_path: str = None):
        """Nạp ảnh an toàn 100% cho mọi định dạng (RGB, Grayscale, RGBA, LA, Palette, CMYK)."""
        try:
            pil_img = Image.open(path)
            # Chuẩn hóa về RGB nếu có kênh trong suốt hoặc bảng màu
            if pil_img.mode in ("RGBA", "LA", "P", "PA", "CMYK"):
                pil_img = pil_img.convert("RGB")
                
            arr = np.array(pil_img)
            self.orig_image = arr
            self.processed_image = arr.copy()

            # Nếu có ảnh sạch để đối chiếu định lượng (Ground Truth)
            if clean_reference_path and os.path.exists(clean_reference_path):
                self.clean_ground_truth = np.array(Image.open(clean_reference_path).convert(pil_img.mode))
            else:
                self.clean_ground_truth = None

            h, w = arr.shape[:2]
            mode_str = "RGB" if arr.ndim == 3 else "Grayscale"
            self.card_orig.set_image(self.orig_image, f"{w}x{h} | {mode_str}")
            self.card_proc.set_image(self.processed_image)

            # Chạy pipeline lần đầu
            self._trigger_pipeline()

        except Exception as e:
            QMessageBox.critical(self, "Lỗi đọc tệp", f"Không thể nạp tệp ảnh:\n{str(e)}")

    def _on_open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh cần khôi phục", "",
            "Tệp hình ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.webp);;Tất cả tệp (*.*)"
        )
        if filepath:
            self.load_image_from_path(filepath)

    def _on_load_sample(self):
        """Mở menu hoặc chọn ảnh mẫu thực tế."""
        sample_path = os.path.join("data", "old", "vietnam_vintage_portrait.jpg")
        if not os.path.exists(sample_path):
            sample_path = os.path.join("data", "old", "sample_simulated_old.png")
            
        clean_ref = os.path.join("data", "clean", "sample_clean_portrait.png")
        self.load_image_from_path(sample_path, clean_reference_path=clean_ref)

    def _on_save_file(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Chưa có ảnh", "Vui lòng mở ảnh trước khi lưu.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Lưu kết quả phục hồi", "anh_da_phuc_hoi.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp)"
        )
        if filepath:
            try:
                Image.fromarray(self.processed_image).save(filepath)
                QMessageBox.information(self, "Thành công", f"Đã lưu ảnh kết quả vào:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi lưu file", f"Không thể lưu tệp:\n{str(e)}")

    # --------------------------------------------------------
    # ĐIỀU KHIỂN PIPELINE VÀ THREAD WORKER
    # --------------------------------------------------------
    def _collect_stages(self) -> list:
        return [
            self.card_denoise.get_stage(),
            self.card_scratch.get_stage(),
            self.card_contrast.get_stage(),
            self.card_sharpen.get_stage(),
        ]

    def _trigger_pipeline(self):
        """Chạy lại toàn bộ pipeline tuần tự từ ảnh gốc ban đầu."""
        if self.orig_image is None:
            return

        stages = self._collect_stages()
        
        # Cập nhật Badge tên các bước đang kích hoạt
        active_names = []
        if self.card_denoise.chk_enable.isChecked():
            active_names.append("Khử nhiễu")
        if self.card_scratch.chk_enable.isChecked():
            active_names.append("Vết xước")
        if self.card_contrast.chk_enable.isChecked():
            active_names.append("Tương phản")
        if self.card_sharpen.chk_enable.isChecked():
            active_names.append("Làm nét")
            
        badge_text = " + ".join(active_names) if active_names else "Gốc (Chưa lọc)"
        self.card_proc.set_badge_text(badge_text)

        # Chạy trong luồng nền QThread để không đơ UI
        self.worker.process(self.orig_image, stages)

    def _on_worker_result(self, result_image: np.ndarray, elapsed_ms: float):
        """Nhận kết quả từ luồng nền và cập nhật giao diện."""
        self.processed_image = result_image
        self.card_proc.set_image(self.processed_image)

        # Nếu có ảnh sạch để đối chiếu, tính e_rms và SNR_ms
        if self.clean_ground_truth is not None and self.clean_ground_truth.shape == self.processed_image.shape:
            rms = calculate_rms_error(self.processed_image, self.clean_ground_truth)
            snr = calculate_snr_ms(self.processed_image, self.clean_ground_truth)
            self.lbl_rms_val.setText(f"{rms:.2f}")
            snr_str = f"{snr:.2f} dB" if snr < 999 else "∞"
            self.lbl_snr_val.setText(snr_str)
        else:
            self.lbl_rms_val.setText("--")
            self.lbl_snr_val.setText("--")

    def _on_worker_error(self, err_msg: str):
        print(f"Lỗi Pipeline: {err_msg}")

    def _on_auto_restore(self):
        """Nút Khôi phục tự động: Chẩn đoán tổn thương ảnh và tự bật các bước phù hợp."""
        if self.orig_image is None:
            return

        stages = diagnose(self.orig_image)
        for s in stages:
            if s.name == "denoise":
                self.card_denoise.set_stage(s)
            elif s.name == "scratch":
                self.card_scratch.set_stage(s)
            elif s.name == "contrast":
                self.card_contrast.set_stage(s)
            elif s.name == "sharpen":
                self.card_sharpen.set_stage(s)

        # Kích hoạt chạy lại pipeline
        self._trigger_pipeline()
