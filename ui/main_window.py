"""
Module ui/main_window.py
Cửa sổ chính: Khôi phục & nâng cao chất lượng ảnh cũ.

Bố cục:
  - Thanh trên: Mở ảnh | Ảnh mẫu ▾ | Tạo ảnh cũ mô phỏng ............ Lưu kết quả
  - Trái: Ảnh gốc và Kết quả đặt cạnh nhau (mỗi ảnh có histogram); hộp "Hiển thị" cho phép
          xem ảnh sau từng bước hoặc mặt nạ vết xước. Bên dưới là bảng chỉ số PSNR / e_rms / SNR
          trước và sau (khi có ảnh sạch để đối chiếu).
  - Phải: nút Khôi phục tự động + lời giải thích chẩn đoán, rồi 4 thẻ của quy trình cố định.

Xử lý:
  - Ảnh lớn được thu nhỏ (trung bình khối) để xem trước mượt; khi Lưu, quy trình chạy lại
    trên ảnh gốc đủ độ phân giải.
  - Mọi lần chỉnh đều chạy lại từ ảnh gốc trong luồng nền (ui/worker.py).
"""
import html
import os

import numpy as np
from PIL import Image, ImageOps

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, QMenu,
    QFileDialog, QMessageBox, QFrame, QScrollArea, QSplitter, QStatusBar, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer

from core.pipeline import Stage, run_pipeline, diagnose_with_report, STAGE_TITLES
from core.metrics import calculate_rms_error, calculate_snr_ms, calculate_psnr
from core.degrade import simulate_old_photo
from core.resize import downscale_factor, block_downscale
from ui.image_pair import ImageCard
from ui.stage_card import StageCard
from ui.worker import PipelineWorker, TaskWorker
from ui import theme

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PREVIEW_MAX_SIDE = 900

SAMPLES = [
    ("Chân dung cũ thật (không có ảnh đối chiếu)",
     os.path.join("old", "vietnam_vintage_portrait.jpg"), None),
    ("Ảnh cũ mô phỏng (có ảnh sạch để đo PSNR)",
     os.path.join("old", "sample_simulated_old.png"), os.path.join("clean", "sample_clean_portrait.png")),
    ("Ảnh tương phản rất thấp",
     os.path.join("old", "sample_low_contrast.png"), None),
]


def load_image_file(path: str) -> np.ndarray:
    """Đọc ảnh bằng Pillow và chuẩn hóa về uint8 xám (H, W) hoặc RGB (H, W, 3)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # xoay đúng chiều theo EXIF của ảnh chụp điện thoại
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.array(img).astype(np.float64)
        mx = arr.max() if arr.size else 1.0
        scale = 255.0 / (65535.0 if mx > 255 else 255.0)
        return np.clip(np.round(arr * scale), 0, 255).astype(np.uint8)
    if img.mode == "L":
        return np.array(img, dtype=np.uint8)
    if img.mode in ("1", "P", "PA", "LA", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "RGBX"):
        has_color = img.mode not in ("1", "LA")
        img = img.convert("RGB" if has_color else "L")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    # Ảnh RGB nhưng thực chất là ảnh xám (3 kênh bằng nhau) -> đưa về 1 kênh cho nhanh và gọn
    if arr.ndim == 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
        arr = arr[..., 0].copy()
    return arr


def match_reference(ref: np.ndarray, target_shape) -> np.ndarray:
    """Đưa ảnh đối chiếu về cùng số kênh với ảnh đang xử lý; None nếu khác kích thước."""
    if ref is None:
        return None
    if ref.shape[:2] != tuple(target_shape[:2]):
        return None
    if len(target_shape) == 2 and ref.ndim == 3:
        g = 0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]
        return np.clip(np.round(g), 0, 255).astype(np.uint8)
    if len(target_shape) == 3 and ref.ndim == 2:
        return np.repeat(ref[..., None], 3, axis=2)
    return ref


def mask_overlay(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tô đỏ các điểm thuộc mặt nạ lên ảnh để quan sát vùng được phát hiện."""
    rgb = np.repeat(base[..., None], 3, axis=2) if base.ndim == 2 else base.copy()
    rgb = (rgb.astype(np.float32) * 0.55).astype(np.uint8)
    m = mask > 0
    rgb[m] = [239, 68, 68]
    return rgb


class MetricBox(QFrame):
    """Ô hiển thị một chỉ số: giá trị trước -> sau."""

    def __init__(self, name: str, hint: str, higher_is_better: bool):
        super().__init__()
        self.setObjectName("card")
        self.higher_is_better = higher_is_better
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        t = QLabel(name)
        t.setObjectName("muted")
        t.setToolTip(hint)
        self.val = QLabel("—")
        self.val.setStyleSheet("font-size: 16px; font-weight: 700;")
        lay.addWidget(t)
        lay.addWidget(self.val)

    def set_values(self, before, after, fmt="{:.2f}", unit=""):
        if before is None or after is None:
            self.val.setText("—")
            self.val.setStyleSheet("font-size: 16px; font-weight: 700; color: #94a3b8;")
            return

        def f(v):
            return "∞" if np.isinf(v) else fmt.format(v)

        better = (after > before) if self.higher_is_better else (after < before)
        color = theme.OK if better else (theme.ERR if after != before else theme.TEXT)
        self.val.setText(f"{f(before)} → {f(after)}{unit}")
        self.val.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {color};")


class MainWindow(QMainWindow):
    def __init__(self, autoload_sample: bool = True):
        super().__init__()
        self.setWindowTitle("Khôi phục & nâng cao chất lượng ảnh cũ — BTL Xử lý ảnh (INT13146)")
        self.resize(1360, 820)
        self.setMinimumSize(1040, 660)
        self.setStyleSheet(theme.APP_STYLESHEET)

        # Dữ liệu
        self.full_image = None          # ảnh gốc đủ độ phân giải
        self.full_reference = None      # ảnh sạch đối chiếu (nếu có), cùng kích thước ảnh gốc
        self.preview_image = None       # ảnh gốc đã thu nhỏ để xem trước
        self.preview_reference = None
        self.preview_factor = 1
        self.result_image = None
        self.result_steps = []
        self.result_mask = None
        self.current_path = None
        self._latest_job = 0
        self._export_worker = None

        self.worker = PipelineWorker(self)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)

        # Gom các thay đổi liên tiếp của thanh trượt (60 ms) rồi mới chạy lại
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(60)
        self._debounce.timeout.connect(self._run_pipeline_now)

        self._build_ui()
        self._set_has_image(False)

        if autoload_sample:
            path = os.path.join(DATA_DIR, SAMPLES[0][1])
            if os.path.exists(path):
                self.load_image_from_path(path)

    # ==================================================================
    # Dựng giao diện
    # ==================================================================
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Thanh tiêu đề ----
        header = QFrame()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 12, 20, 12)
        hl.setSpacing(10)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        t1 = QLabel("Khôi phục ảnh cũ")
        t1.setObjectName("appTitle")
        t2 = QLabel("Khử nhiễu · Xóa vết xước · Tăng tương phản · Làm nét — thuật toán tự cài đặt bằng NumPy")
        t2.setObjectName("appSubtitle")
        titles.addWidget(t1)
        titles.addWidget(t2)
        hl.addLayout(titles)
        hl.addStretch()

        self.btn_open = QPushButton("Mở ảnh…")
        self.btn_open.setShortcut("Ctrl+O")
        self.btn_open.setToolTip("Mở ảnh từ máy (Ctrl+O)")
        self.btn_open.clicked.connect(self._on_open_file)
        hl.addWidget(self.btn_open)

        self.btn_sample = QToolButton()
        self.btn_sample.setText("Ảnh mẫu")
        self.btn_sample.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_sample)
        for label, rel, ref in SAMPLES:
            act = menu.addAction(label)
            act.triggered.connect(lambda _=False, r=rel, g=ref: self._load_sample(r, g))
        self.btn_sample.setMenu(menu)
        hl.addWidget(self.btn_sample)

        self.btn_simulate = QPushButton("Tạo ảnh cũ mô phỏng")
        self.btn_simulate.setToolTip(
            "Coi ảnh đang mở là ảnh sạch, tạo bản 'cũ hóa' (bạc màu, nhiễu, vết xước)\n"
            "để đo PSNR trước/sau khi khôi phục.")
        self.btn_simulate.clicked.connect(self._on_simulate)
        hl.addWidget(self.btn_simulate)

        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background: {theme.BORDER};")
        hl.addWidget(sep)

        self.btn_save = QPushButton("Lưu kết quả")
        self.btn_save.setObjectName("primary")
        self.btn_save.setShortcut("Ctrl+S")
        self.btn_save.setToolTip("Chạy lại trên ảnh gốc đủ độ phân giải rồi lưu (Ctrl+S)")
        self.btn_save.clicked.connect(self._on_save_file)
        hl.addWidget(self.btn_save)
        outer.addWidget(header)

        # ---- Thân ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        # Trái: ảnh + chỉ số
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(20, 16, 6, 16)
        ll.setSpacing(12)
        imgs = QHBoxLayout()
        imgs.setSpacing(12)
        self.card_orig = ImageCard("Ảnh gốc", is_result=False)
        self.card_result = ImageCard("Kết quả", is_result=True)
        self.card_result.viewChanged.connect(lambda _: self._show_result_view())
        imgs.addWidget(self.card_orig, 1)
        imgs.addWidget(self.card_result, 1)
        ll.addLayout(imgs, 1)

        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.m_psnr = MetricBox("PSNR so với ảnh sạch (dB)", "Càng cao càng giống ảnh sạch", True)
        self.m_rms = MetricBox("e_rms (Ch.4)", "Sai số căn quân phương, càng thấp càng tốt", False)
        self.m_snr = MetricBox("SNR_ms (dB, Ch.4)", "Tỉ số tín hiệu/nhiễu, càng cao càng tốt", True)
        for m in (self.m_psnr, self.m_rms, self.m_snr):
            metrics.addWidget(m, 1)
        ll.addLayout(metrics)
        self.lbl_metric_hint = QLabel("Chỉ số trước → sau chỉ tính được khi có ảnh sạch để đối chiếu "
                                      "(chọn “Ảnh mẫu › Ảnh cũ mô phỏng” hoặc bấm “Tạo ảnh cũ mô phỏng”).")
        self.lbl_metric_hint.setObjectName("muted")
        self.lbl_metric_hint.setWordWrap(True)
        ll.addWidget(self.lbl_metric_hint)
        splitter.addWidget(left)

        # Phải: điều khiển
        right_inner = QWidget()
        rl = QVBoxLayout(right_inner)
        rl.setContentsMargins(6, 16, 20, 16)
        rl.setSpacing(10)

        auto_row = QHBoxLayout()
        self.btn_auto = QPushButton("Khôi phục tự động")
        self.btn_auto.setObjectName("autoButton")
        self.btn_auto.setCursor(Qt.PointingHandCursor)
        self.btn_auto.setToolTip("Đo mức nhiễu, vết xước, dải xám rồi tự chọn các bước phù hợp")
        self.btn_auto.clicked.connect(self._on_auto_restore)
        self.btn_reset = QPushButton("Đặt lại")
        self.btn_reset.setObjectName("ghost")
        self.btn_reset.setToolTip("Tắt hết các bước, trở về ảnh gốc")
        self.btn_reset.clicked.connect(self._on_reset)
        auto_row.addWidget(self.btn_auto, 1)
        auto_row.addWidget(self.btn_reset)
        rl.addLayout(auto_row)

        self.lbl_report = QLabel("Bấm “Khôi phục tự động” để xem chẩn đoán, hoặc tự bật từng bước bên dưới.")
        self.lbl_report.setObjectName("report")
        self.lbl_report.setWordWrap(True)
        self.lbl_report.setTextFormat(Qt.RichText)
        rl.addWidget(self.lbl_report)

        sec = QLabel("QUY TRÌNH (THỨ TỰ CỐ ĐỊNH)")
        sec.setObjectName("sectionTitle")
        rl.addWidget(sec)

        self.cards = {
            "scratch": StageCard("scratch", 1, "Xóa vết xước, bụi"),
            "denoise": StageCard("denoise", 2, "Khử nhiễu"),
            "contrast": StageCard("contrast", 3, "Sáng và tương phản"),
            "sharpen": StageCard("sharpen", 4, "Làm nét"),
        }
        for card in self.cards.values():
            card.changed.connect(self._schedule_pipeline)
            rl.addWidget(card)
        rl.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(right_inner)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(480)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([940, 420])
        outer.addWidget(splitter, 1)

        # ---- Thanh trạng thái ----
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_status = QLabel("Sẵn sàng")
        self.status.addWidget(self.lbl_status, 1)
        self.lbl_size = QLabel("")
        self.status.addPermanentWidget(self.lbl_size)

    # ==================================================================
    # Nạp ảnh
    # ==================================================================
    def _set_has_image(self, has: bool):
        for w in (self.btn_save, self.btn_auto, self.btn_reset, self.btn_simulate):
            w.setEnabled(has)

    def _set_status(self, text: str, color: str = None):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color};" if color else "")

    def load_image_from_path(self, path: str, reference_path: str = None) -> bool:
        try:
            img = load_image_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Không đọc được ảnh", f"Không thể mở tệp:\n{path}\n\n{e}")
            return False
        ref = None
        if reference_path and os.path.exists(reference_path):
            try:
                ref = match_reference(load_image_file(reference_path), img.shape)
            except Exception:
                ref = None
        self.current_path = path
        self.set_images(img, ref, os.path.basename(path))
        return True

    def set_images(self, image: np.ndarray, reference: np.ndarray = None, name: str = ""):
        """Đặt ảnh gốc (và ảnh sạch đối chiếu nếu có), tạo bản xem trước, chạy lại quy trình."""
        self.full_image = image
        self.full_reference = match_reference(reference, image.shape) if reference is not None else None
        self.preview_factor = downscale_factor(image.shape, PREVIEW_MAX_SIDE)
        self.preview_image = block_downscale(image, self.preview_factor)
        self.preview_reference = (block_downscale(self.full_reference, self.preview_factor)
                                  if self.full_reference is not None else None)
        self.result_image = self.preview_image
        self.result_steps, self.result_mask = [], None

        h, w = image.shape[:2]
        kind = "RGB" if image.ndim == 3 else "xám"
        self.card_orig.set_info(f"{w}×{h} · {kind}")
        self.card_orig.set_image(self.preview_image)
        self.card_result.set_image(self.preview_image)
        self.card_result.set_view_options([("final", "Kết quả cuối")])
        ph, pw = self.preview_image.shape[:2]
        note = f" · xem trước {pw}×{ph} (thu nhỏ ×{self.preview_factor})" if self.preview_factor > 1 else ""
        self.lbl_size.setText(f"{name}  {w}×{h}{note}")
        self.lbl_metric_hint.setVisible(self.preview_reference is None)
        self._set_has_image(True)
        self._run_pipeline_now()

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh cần khôi phục", DATA_DIR,
            "Ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;Tất cả tệp (*.*)")
        if path:
            self.load_image_from_path(path)

    def _load_sample(self, rel: str, ref_rel: str = None):
        path = os.path.join(DATA_DIR, rel)
        if not os.path.exists(path):
            QMessageBox.warning(self, "Thiếu ảnh mẫu",
                                f"Không tìm thấy {path}.\nChạy: python data/create_samples.py")
            return
        ref = os.path.join(DATA_DIR, ref_rel) if ref_rel else None
        self.load_image_from_path(path, ref)

    def _on_simulate(self):
        if self.full_image is None:
            return
        clean = self.full_image
        degraded, _ = simulate_old_photo(
            clean, noise_type="both", noise_level=0.015, gamma_val=1.0,
            num_scratches=max(4, clean.shape[1] // 60), seed=7, fade=0.45)
        self.set_images(degraded, clean, "ảnh mô phỏng")
        self._set_status("Đã tạo ảnh cũ mô phỏng từ ảnh hiện tại. Ảnh ban đầu được dùng làm ảnh sạch để đo PSNR.")

    # ==================================================================
    # Chạy quy trình
    # ==================================================================
    def _collect_stages(self):
        return [card.get_stage() for card in self.cards.values()]

    def _schedule_pipeline(self):
        if self.preview_image is not None:
            self._debounce.start()

    def _run_pipeline_now(self):
        if self.preview_image is None:
            return
        self._set_status("Đang xử lý…", theme.ACCENT)
        self._latest_job = self.worker.process(self.preview_image, self._collect_stages())

    def _on_result(self, job_id, image, steps, mask, elapsed_ms):
        if job_id != self._latest_job:
            return  # kết quả của lần chỉnh cũ, bỏ qua
        self.result_image = image
        self.result_steps = steps
        self.result_mask = mask

        options = [("final", "Kết quả cuối")]
        for i, (name, _) in enumerate(steps, start=1):
            options.append((f"step:{i - 1}", f"Sau bước: {STAGE_TITLES.get(name, name)}"))
        if mask is not None:
            options.append(("mask", f"Mặt nạ vết xước ({np.mean(mask > 0) * 100:.1f}%)"))
        self.card_result.set_view_options(options)
        self._show_result_view()
        self._update_metrics()

        n = len(steps)
        msg = f"Xong: {n} bước, {elapsed_ms:.0f} ms" if n else "Chưa bật bước nào: kết quả = ảnh gốc"
        self._set_status(msg, theme.OK)

    def _show_result_view(self):
        if self.result_image is None:
            return
        view = self.card_result.current_view()
        if view == "mask" and self.result_mask is not None:
            self.card_result.set_image(mask_overlay(self.preview_image, self.result_mask))
            self.card_result.set_info("")
            return
        if view.startswith("step:"):
            idx = int(view.split(":")[1])
            if 0 <= idx < len(self.result_steps):
                self.card_result.set_image(self.result_steps[idx][1])
                return
        self.card_result.set_image(self.result_image)

    def _update_metrics(self):
        ref = self.preview_reference
        if ref is None or self.result_image is None or ref.shape != self.result_image.shape:
            for m in (self.m_psnr, self.m_rms, self.m_snr):
                m.set_values(None, None)
            return
        src, out = self.preview_image, self.result_image
        self.m_psnr.set_values(calculate_psnr(src, ref), calculate_psnr(out, ref))
        self.m_rms.set_values(calculate_rms_error(src, ref), calculate_rms_error(out, ref))
        self.m_snr.set_values(calculate_snr_ms(src, ref), calculate_snr_ms(out, ref))

    def _on_error(self, job_id, msg):
        if job_id != self._latest_job:
            return
        self._set_status(f"Lỗi xử lý: {msg}", theme.ERR)

    # ==================================================================
    # Tự động / đặt lại
    # ==================================================================
    def _on_auto_restore(self):
        if self.preview_image is None:
            return
        stages, report, _ = diagnose_with_report(self.preview_image)
        for st in stages:
            self.cards[st.name].set_stage(st)
        items = "".join(f"<li style='margin-bottom:4px'>{html.escape(line)}</li>" for line in report)
        self.lbl_report.setText(f"<b>Chẩn đoán</b><ul style='margin:4px 0 0 -18px'>{items}</ul>")
        self._run_pipeline_now()

    def _on_reset(self):
        for name, card in self.cards.items():
            card.set_stage(Stage(name, False, card.get_stage().method, card.get_stage().params))
        self.lbl_report.setText("Đã tắt mọi bước. Bấm “Khôi phục tự động” hoặc tự bật từng bước.")
        self._run_pipeline_now()

    # ==================================================================
    # Lưu
    # ==================================================================
    def _on_save_file(self):
        if self.full_image is None:
            return
        base = os.path.splitext(os.path.basename(self.current_path or "anh"))[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh đã khôi phục", os.path.join(os.path.expanduser("~"), f"{base}_khoi_phuc.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".png"
        stages = self._collect_stages()
        self.btn_save.setEnabled(False)
        self._set_status("Đang xử lý ảnh ở độ phân giải gốc để lưu (ảnh lớn có thể mất vài chục giây)…", theme.ACCENT)

        def job(img, st, out_path):
            res = run_pipeline(img, st)
            Image.fromarray(res).save(out_path)
            return out_path

        self._export_worker = TaskWorker(job, self.full_image, stages, path, parent=self)
        self._export_worker.finished_ok.connect(self._on_saved)
        self._export_worker.failed.connect(self._on_save_failed)
        self._export_worker.start()

    def _on_saved(self, path):
        self.btn_save.setEnabled(True)
        self._set_status(f"Đã lưu: {path}", theme.OK)

    def _on_save_failed(self, msg):
        self.btn_save.setEnabled(True)
        self._set_status("Lưu thất bại", theme.ERR)
        QMessageBox.critical(self, "Lưu thất bại", msg)

    # ==================================================================
    def closeEvent(self, event):
        self._debounce.stop()
        self.worker.wait(3000)
        if self._export_worker is not None:
            self._export_worker.wait(10000)
        super().closeEvent(event)
