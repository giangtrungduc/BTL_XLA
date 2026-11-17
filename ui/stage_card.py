"""
Module ui/stage_card.py
Component giao diện thẻ bước khôi phục (Stage Card).
Mỗi thẻ đại diện cho một bước trong quy trình:
  1. Khử nhiễu
  2. Vết xước
  3. Sáng và tương phản
  4. Làm nét
Thiết kế tinh tế, trực quan bám sát theo ảnh mock-up.
"""
from typing import Dict, Any
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox,
    QLabel, QSlider, QWidget
)
from PySide6.QtCore import Qt, Signal

from core.pipeline import Stage


class StageCard(QFrame):
    """Thẻ điều khiển một giai đoạn khôi phục."""
    changed = Signal()

    def __init__(self, stage_name: str, title: str, parent=None):
        super().__init__(parent)
        self.stage_name = stage_name
        self.title = title
        
        # Style thẻ card viền bo tròn nhẹ
        self.setStyleSheet("""
            StageCard {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
            QCheckBox {
                font-weight: bold;
                font-size: 13px;
                color: #0f172a;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QComboBox {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 11px;
                color: #1e293b;
            }
            QComboBox::drop-down {
                border: none;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #e2e8f0;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #0284c7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #0284c7;
                width: 14px;
                height: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
        """)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 10, 12, 10)
        self.layout.setSpacing(8)

        # 1. Header có Checkbox
        self.chk_enable = QCheckBox(self.title)
        self.chk_enable.setChecked(True)
        self.chk_enable.toggled.connect(self._on_changed)
        self.layout.addWidget(self.chk_enable)

        # 2. Khung nội dung các điều khiển
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(6)
        self.layout.addWidget(self.content_widget)

        self._setup_controls()

    def _setup_controls(self):
        """Khởi tạo các control tùy theo stage."""
        if self.stage_name == "denoise":
            row = QHBoxLayout()
            self.cbo_method = QComboBox()
            self.cbo_method.addItems([
                "Trung vị (Median)",
                "Trung bình số học (Mean)",
                "Trung bình hình học (Geometric)",
                "Trung bình điều hòa (Harmonic)",
                "Điều hòa ngược (Contraharmonic Q)",
                "Lọc cực đại (Max)",
                "Lọc cực tiểu (Min)",
                "Lọc điểm giữa (Midpoint)",
                "Cắt tỉa Alpha (Alpha-trimmed)",
                "K lân cận gần nhất (K-Nearest)",
                "Lọc Gaussian"
            ])
            self.cbo_method.currentIndexChanged.connect(self._on_changed)
            row.addWidget(self.cbo_method, 7)

            self.cbo_ksize = QComboBox()
            self.cbo_ksize.addItems(["3 x 3", "5 x 5", "7 x 7"])
            self.cbo_ksize.currentIndexChanged.connect(self._on_changed)
            row.addWidget(self.cbo_ksize, 3)
            self.content_layout.addLayout(row)

        elif self.stage_name == "scratch":
            row_slider = QHBoxLayout()
            self.lbl_slider = QLabel("Độ nhạy: 50%")
            self.lbl_slider.setStyleSheet("font-size: 11px; color: #475569;")
            
            self.slider_sens = QSlider(Qt.Horizontal)
            self.slider_sens.setRange(10, 90)
            self.slider_sens.setValue(50)
            self.slider_sens.valueChanged.connect(self._on_scratch_slider_change)
            
            row_slider.addWidget(self.lbl_slider)
            row_slider.addWidget(self.slider_sens)
            self.content_layout.addLayout(row_slider)

        elif self.stage_name == "contrast":
            self.cbo_method = QComboBox()
            self.cbo_method.addItems([
                "Cân bằng histogram (CDF)",
                "Biến đổi Gamma (s = c·r^γ)",
                "Biến đổi Log (s = c·log(1+r))",
                "Biến đổi Âm bản (255 - r)",
                "Kéo giãn từng đoạn (Contrast Stretching)"
            ])
            self.cbo_method.currentIndexChanged.connect(self._on_changed)
            self.content_layout.addWidget(self.cbo_method)

            row_gamma = QHBoxLayout()
            self.lbl_gamma = QLabel("γ: 0.80")
            self.lbl_gamma.setStyleSheet("font-size: 11px; color: #475569;")
            self.slider_gamma = QSlider(Qt.Horizontal)
            self.slider_gamma.setRange(20, 250)
            self.slider_gamma.setValue(80)
            self.slider_gamma.valueChanged.connect(self._on_gamma_slider_change)
            row_gamma.addWidget(self.lbl_gamma)
            row_gamma.addWidget(self.slider_gamma)
            self.content_layout.addLayout(row_gamma)

        elif self.stage_name == "sharpen":
            self.cbo_method = QComboBox()
            self.cbo_method.addItems([
                "Laplacian 8 lân cận",
                "Laplacian 4 lân cận",
                "Sobel Gradient",
                "Chế độ kết hợp nâng cao (Ch.6)"
            ])
            self.cbo_method.currentIndexChanged.connect(self._on_changed)
            self.content_layout.addWidget(self.cbo_method)

            row_k = QHBoxLayout()
            self.lbl_k = QLabel("k: 0.40")
            self.lbl_k.setStyleSheet("font-size: 11px; color: #475569;")
            self.slider_k = QSlider(Qt.Horizontal)
            self.slider_k.setRange(10, 150)
            self.slider_k.setValue(40)
            self.slider_k.valueChanged.connect(self._on_k_slider_change)
            row_k.addWidget(self.lbl_k)
            row_k.addWidget(self.slider_k)
            self.content_layout.addLayout(row_k)

    def _on_scratch_slider_change(self, val):
        self.lbl_slider.setText(f"Độ nhạy: {val}%")
        self._on_changed()

    def _on_gamma_slider_change(self, val):
        gamma_float = val / 100.0
        self.lbl_gamma.setText(f"γ: {gamma_float:.2f}")
        self._on_changed()

    def _on_k_slider_change(self, val):
        k_float = val / 100.0
        self.lbl_k.setText(f"k: {k_float:.2f}")
        self._on_changed()

    def _on_changed(self):
        self.content_widget.setEnabled(self.chk_enable.isChecked())
        self.changed.emit()

    def get_stage(self) -> Stage:
        """Trích xuất trạng thái hiện tại thành đối tượng Stage."""
        enabled = self.chk_enable.isChecked()
        params = {}
        method = "default"

        if self.stage_name == "denoise":
            method_txt = self.cbo_method.currentText()
            method = method_txt.lower()
            k_size = int(self.cbo_ksize.currentText()[0])
            params["kernel_size"] = k_size
            params["pad_mode"] = "mirror"

        elif self.stage_name == "scratch":
            method = "median_fill"
            params["sensitivity"] = self.slider_sens.value() / 100.0
            params["fill_method"] = "median"
            params["kernel_size"] = 7

        elif self.stage_name == "contrast":
            method_txt = self.cbo_method.currentText()
            method = method_txt.lower()
            params["gamma"] = self.slider_gamma.value() / 100.0
            params["eq_mode"] = "stretch_to_zero"

        elif self.stage_name == "sharpen":
            method_txt = self.cbo_method.currentText()
            method = method_txt.lower()
            params["k"] = self.slider_k.value() / 100.0
            params["pad_mode"] = "mirror"

        return Stage(name=self.stage_name, enabled=enabled, method=method, params=params)

    def set_stage(self, stage: Stage):
        """Cập nhật giá trị vào UI từ đối tượng Stage."""
        self.blockSignals(True)
        self.chk_enable.setChecked(stage.enabled)
        
        if self.stage_name == "denoise":
            k_size = stage.params.get("kernel_size", 3)
            idx = 0 if k_size == 3 else (1 if k_size == 5 else 2)
            self.cbo_ksize.setCurrentIndex(idx)
            # Chọn method
            for i in range(self.cbo_method.count()):
                if stage.method in self.cbo_method.itemText(i).lower():
                    self.cbo_method.setCurrentIndex(i)
                    break

        elif self.stage_name == "scratch":
            sens = int(stage.params.get("sensitivity", 0.5) * 100)
            self.slider_sens.setValue(sens)
            self.lbl_slider.setText(f"Độ nhạy: {sens}%")

        elif self.stage_name == "contrast":
            if "gamma" in stage.params:
                g = int(stage.params["gamma"] * 100)
                self.slider_gamma.setValue(g)
                self.lbl_gamma.setText(f"γ: {g/100.0:.2f}")
            for i in range(self.cbo_method.count()):
                if stage.method in self.cbo_method.itemText(i).lower():
                    self.cbo_method.setCurrentIndex(i)
                    break

        elif self.stage_name == "sharpen":
            if "k" in stage.params:
                k_val = int(stage.params["k"] * 100)
                self.slider_k.setValue(k_val)
                self.lbl_k.setText(f"k: {k_val/100.0:.2f}")
            for i in range(self.cbo_method.count()):
                if stage.method in self.cbo_method.itemText(i).lower():
                    self.cbo_method.setCurrentIndex(i)
                    break

        self.content_widget.setEnabled(self.chk_enable.isChecked())
        self.blockSignals(False)
