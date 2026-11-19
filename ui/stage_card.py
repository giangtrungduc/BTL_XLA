"""
Module ui/stage_card.py
Thẻ điều khiển một bước trong quy trình khôi phục:
  1. Vết xước  2. Khử nhiễu  3. Sáng & tương phản  4. Làm nét

Mỗi thẻ gồm: công tắc bật/tắt, chọn phương pháp, thanh trượt tham số (chỉ hiện khi phương pháp cần),
và một dòng công thức ngắn để người xem hiểu bước đó làm gì.
Giao diện chỉ truyền KHÓA phương pháp chuẩn (vd. 'median', 'stretch') cho core/pipeline.py.
"""
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel, QSlider, QWidget
)
from PySide6.QtCore import Qt, Signal, QSignalBlocker

from core.pipeline import Stage


# (khóa, tên hiển thị, công thức / mô tả ngắn)
DENOISE_OPTIONS = [
    ("median", "Trung vị (Median)", "f̂(x,y) = median{ g(s,t) }  — tốt nhất cho nhiễu muối tiêu"),
    ("mean", "Trung bình số học", "f̂(x,y) = (1/mn) Σ g(s,t)  — làm mịn đều, mờ biên"),
    ("gaussian", "Gauss", "f̂ = Σ w(s,t) g(s,t),  w ∝ exp(−(s²+t²)/2σ²)  — nhiễu hạt"),
    ("alpha_trimmed", "Trung bình cắt tỉa α", "Bỏ d/2 giá trị nhỏ nhất và lớn nhất rồi lấy trung bình — nhiễu hỗn hợp"),
    ("k_nearest", "Trung bình k lân cận gần nhất", "Trung bình k điểm có mức xám gần điểm tâm nhất — giữ biên"),
    ("geometric", "Trung bình hình học", "f̂ = [Π g(s,t)]^(1/mn)"),
    ("harmonic", "Trung bình điều hòa", "f̂ = mn / Σ 1/g(s,t)  — tốt với nhiễu muối, kém với nhiễu tiêu"),
    ("contraharmonic", "Điều hòa ngược (Q)", "f̂ = Σ g^(Q+1) / Σ g^Q  — Q>0 khử tiêu, Q<0 khử muối"),
    ("midpoint", "Điểm giữa (Midpoint)", "f̂ = ½ [max g(s,t) + min g(s,t)]"),
    ("max", "Cực đại (Max)", "f̂ = max{ g(s,t) }  — khử nhiễu tiêu"),
    ("min", "Cực tiểu (Min)", "f̂ = min{ g(s,t) }  — khử nhiễu muối"),
]

CONTRAST_OPTIONS = [
    ("stretch", "Kéo giãn tương phản", "Ánh xạ tuyến tính [p%, 100−p%] → [0, 255]"),
    ("equalize", "Cân bằng histogram", "s_k = 255 · (CDF(r_k) − CDF_min) / (MN − CDF_min)"),
    ("gamma", "Biến đổi Gamma", "s = 255 · (r/255)^γ   — γ<1 làm sáng, γ>1 làm đậm"),
    ("log", "Biến đổi Log", "s = c · log(1 + r)   — kéo giãn vùng tối"),
    ("negative", "Âm bản", "s = 255 − r   — dùng cho phim âm bản"),
]

SHARPEN_OPTIONS = [
    ("combined", "Laplacian có trọng số Sobel", "g = f − k · W · ∇²f,  W = |∇f| làm mịn 5×5, chuẩn hóa [0,1]"),
    ("laplacian_4", "Laplacian 4 lân cận", "g = f − k · ∇²f   (mặt nạ [0 1 0; 1 −4 1; 0 1 0])"),
    ("laplacian_8", "Laplacian 8 lân cận", "g = f − k · ∇²f   (mặt nạ [1 1 1; 1 −8 1; 1 1 1])"),
]

POLARITY_OPTIONS = [("bright", "Vệt sáng"), ("dark", "Vệt tối"), ("both", "Cả hai")]
FILL_OPTIONS = [("median", "Trung vị điểm lành"), ("k_nearest", "k lân cận gần nhất")]
KERNEL_OPTIONS = [(3, "3 × 3"), (5, "5 × 5"), (7, "7 × 7")]

SCRATCH_FORMULA = "Mặt nạ = Otsu(g − median₇ₓ₇(g)) → đóng → lọc thành phần liên thông → điền từ mép vào"


class ParamSlider(QWidget):
    """Thanh trượt có nhãn, lưu giá trị thực = giá trị nguyên × scale."""
    valueChanged = Signal()

    def __init__(self, label: str, lo: float, hi: float, value: float, step: float, fmt: str = "{:.2f}"):
        super().__init__()
        self.step = step
        self.fmt = fmt
        self.label_text = label
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.lbl = QLabel()
        self.lbl.setObjectName("muted")
        self.lbl.setMinimumWidth(92)
        self.slider = QSlider(Qt.Horizontal)
        self.set_range(lo, hi)
        lay.addWidget(self.lbl)
        lay.addWidget(self.slider, 1)
        self.slider.valueChanged.connect(self._on_change)
        self.set_value(value)

    def set_range(self, lo: float, hi: float):
        with QSignalBlocker(self.slider):
            self.slider.setRange(int(round(lo / self.step)), int(round(hi / self.step)))

    def set_label(self, label: str, fmt: str = None):
        self.label_text = label
        if fmt:
            self.fmt = fmt
        self._update_label()

    def value(self) -> float:
        return self.slider.value() * self.step

    def set_value(self, v: float):
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(round(v / self.step)))
        self._update_label()

    def _update_label(self):
        self.lbl.setText(f"{self.label_text}: {self.fmt.format(self.value())}")

    def _on_change(self, _):
        self._update_label()
        self.valueChanged.emit()


class StageCard(QFrame):
    """Thẻ điều khiển một bước khôi phục."""
    changed = Signal()

    def __init__(self, stage_name: str, number: int, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.stage_name = stage_name
        self.title = title

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.badge = QLabel(str(number))
        self.badge.setObjectName("stepBadge")
        self.chk_enable = QCheckBox(title)
        self.chk_enable.setChecked(False)
        self.chk_enable.toggled.connect(self._on_toggled)
        head.addWidget(self.badge)
        head.addWidget(self.chk_enable, 1)
        root.addLayout(head)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(7)
        root.addWidget(self.content)

        self.lbl_formula = QLabel()
        self.lbl_formula.setObjectName("formula")
        self.lbl_formula.setWordWrap(True)

        self._build_controls()
        self.content_layout.addWidget(self.lbl_formula)
        self._refresh_visibility()
        self._apply_enabled_look()

    # ------------------------------------------------------------------
    def _combo(self, options, with_desc=False):
        cbo = QComboBox()
        for opt in options:
            cbo.addItem(opt[1], opt[0])
        cbo.currentIndexChanged.connect(self._on_method_changed)
        return cbo

    def _build_controls(self):
        n = self.stage_name
        if n == "scratch":
            row = QHBoxLayout()
            self.cbo_polarity = self._combo(POLARITY_OPTIONS)
            self.cbo_fill = self._combo(FILL_OPTIONS)
            row.addWidget(self.cbo_polarity, 1)
            row.addWidget(self.cbo_fill, 1)
            self.content_layout.addLayout(row)
            self.sl_main = ParamSlider("Độ nhạy", 0.1, 0.9, 0.5, 0.05, "{:.0%}")
            self.sl_main.valueChanged.connect(self._emit)
            self.content_layout.addWidget(self.sl_main)
            self.lbl_formula.setText(SCRATCH_FORMULA)

        elif n == "denoise":
            row = QHBoxLayout()
            self.cbo_method = self._combo(DENOISE_OPTIONS)
            self.cbo_ksize = self._combo(KERNEL_OPTIONS)
            row.addWidget(self.cbo_method, 3)
            row.addWidget(self.cbo_ksize, 1)
            self.content_layout.addLayout(row)
            self.sl_main = ParamSlider("σ", 0.3, 3.0, 1.0, 0.1)
            self.sl_main.valueChanged.connect(self._emit)
            self.content_layout.addWidget(self.sl_main)

        elif n == "contrast":
            self.cbo_method = self._combo(CONTRAST_OPTIONS)
            self.content_layout.addWidget(self.cbo_method)
            self.sl_main = ParamSlider("γ", 0.2, 3.0, 1.0, 0.05)
            self.sl_main.valueChanged.connect(self._emit)
            self.content_layout.addWidget(self.sl_main)

        elif n == "sharpen":
            self.cbo_method = self._combo(SHARPEN_OPTIONS)
            self.content_layout.addWidget(self.cbo_method)
            self.sl_main = ParamSlider("k", 0.1, 1.5, 0.4, 0.05)
            self.sl_main.valueChanged.connect(self._emit)
            self.content_layout.addWidget(self.sl_main)

        # Giá trị riêng cho từng phương pháp, để khi đổi qua lại không mất giá trị đã chỉnh
        self._param_memory = {}

    # ------------------------------------------------------------------
    def _method(self) -> str:
        if self.stage_name == "scratch":
            return "median_fill"
        return self.cbo_method.currentData()

    def _ksize(self) -> int:
        return int(self.cbo_ksize.currentData()) if self.stage_name == "denoise" else 3

    def _slider_spec(self):
        """(nhãn, min, max, mặc định, bước, định dạng) cho thanh trượt theo phương pháp hiện tại, None nếu ẩn."""
        n, m = self.stage_name, self._method()
        if n == "scratch":
            return ("Độ nhạy", 0.1, 0.9, 0.5, 0.05, "{:.0%}")
        if n == "denoise":
            k = self._ksize()
            if m == "gaussian":
                return ("σ", 0.3, 3.0, 1.0, 0.1, "{:.1f}")
            if m == "contraharmonic":
                return ("Q", -3.0, 3.0, 1.5, 0.1, "{:+.1f}")
            if m == "alpha_trimmed":
                return ("d (số điểm bỏ)", 0, k * k - 1, min(4, k * k - 1), 2, "{:.0f}")
            if m == "k_nearest":
                return ("k điểm", 1, k * k, max(3, (k * k) // 2), 1, "{:.0f}")
            return None
        if n == "contrast":
            if m == "gamma":
                return ("γ", 0.2, 3.0, 1.0, 0.05, "{:.2f}")
            if m == "stretch":
                return ("Bỏ 2 đầu (%)", 0.0, 5.0, 1.0, 0.5, "{:.1f}")
            return None
        if n == "sharpen":
            return ("k", 0.1, 1.5, 0.4, 0.05, "{:.2f}")
        return None

    def _formula_for_method(self) -> str:
        m = self._method()
        table = {"denoise": DENOISE_OPTIONS, "contrast": CONTRAST_OPTIONS, "sharpen": SHARPEN_OPTIONS}
        if self.stage_name == "scratch":
            return SCRATCH_FORMULA
        for key, _, desc in table[self.stage_name]:
            if key == m:
                return desc
        return ""

    def _refresh_visibility(self, restore_value: bool = True):
        spec = self._slider_spec()
        if spec is None:
            self.sl_main.hide()
        else:
            label, lo, hi, default, step, fmt = spec
            self.sl_main.step = step
            self.sl_main.set_range(lo, hi)
            self.sl_main.set_label(label, fmt)
            if restore_value:
                key = (self._method(), self._ksize())
                self.sl_main.set_value(self._param_memory.get(key, default))
            self.sl_main.show()
        self.lbl_formula.setText(self._formula_for_method())

    # ------------------------------------------------------------------
    def _on_method_changed(self, _=None):
        self._refresh_visibility()
        self._emit()

    def _on_toggled(self, _):
        self._apply_enabled_look()
        self.changed.emit()

    def _emit(self):
        if self._slider_spec() is not None:
            self._param_memory[(self._method(), self._ksize())] = self.sl_main.value()
        self.changed.emit()

    def _apply_enabled_look(self):
        on = self.chk_enable.isChecked()
        self.content.setEnabled(on)
        self.badge.setObjectName("stepBadge" if on else "stepBadgeOff")
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)

    # ------------------------------------------------------------------
    def get_stage(self) -> Stage:
        enabled = self.chk_enable.isChecked()
        m = self._method()
        params = {}
        n = self.stage_name
        if n == "scratch":
            params = {
                "sensitivity": round(self.sl_main.value(), 3),
                "polarity": self.cbo_polarity.currentData(),
                "fill_method": self.cbo_fill.currentData(),
                "kernel_size": 5,
            }
        elif n == "denoise":
            k = self._ksize()
            params = {"kernel_size": k, "pad_mode": "mirror"}
            v = self.sl_main.value()
            if m == "gaussian":
                params["sigma"] = round(v, 2)
            elif m == "contraharmonic":
                params["Q"] = round(v, 2)
            elif m == "alpha_trimmed":
                params["d"] = int(min(round(v), k * k - 1))
            elif m == "k_nearest":
                params["k"] = int(max(1, min(round(v), k * k)))
        elif n == "contrast":
            v = self.sl_main.value()
            if m == "gamma":
                params["gamma"] = round(v, 3)
            elif m == "stretch":
                params["clip"] = round(v, 2)
            elif m == "equalize":
                params["eq_mode"] = "stretch_to_zero"
        elif n == "sharpen":
            params["k"] = round(self.sl_main.value(), 3)
        return Stage(name=n, enabled=enabled, method=m, params=params)

    def set_stage(self, stage: Stage):
        """Nạp cấu hình từ Stage (vd. từ chẩn đoán tự động) mà không phát tín hiệu trung gian."""
        n = self.stage_name
        blockers = [QSignalBlocker(self.chk_enable), QSignalBlocker(self.sl_main.slider)]
        combos = []
        if n == "scratch":
            combos = [self.cbo_polarity, self.cbo_fill]
        elif n == "denoise":
            combos = [self.cbo_method, self.cbo_ksize]
        else:
            combos = [self.cbo_method]
        blockers += [QSignalBlocker(c) for c in combos]

        self.chk_enable.setChecked(bool(stage.enabled))
        p = stage.params or {}

        def _select(cbo, key):
            idx = cbo.findData(key)
            if idx >= 0:
                cbo.setCurrentIndex(idx)

        if n == "scratch":
            _select(self.cbo_polarity, p.get("polarity", "bright"))
            _select(self.cbo_fill, p.get("fill_method", "median"))
        elif n == "denoise":
            from core.pipeline import _norm_denoise
            _select(self.cbo_method, _norm_denoise(stage.method))
            _select(self.cbo_ksize, int(p.get("kernel_size", 3)))
        elif n == "contrast":
            from core.pipeline import _norm_contrast
            _select(self.cbo_method, _norm_contrast(stage.method))
        elif n == "sharpen":
            from core.pipeline import _norm_sharpen
            _select(self.cbo_method, _norm_sharpen(stage.method))

        self._refresh_visibility(restore_value=False)
        spec = self._slider_spec()
        if spec is not None:
            m = self._method()
            value_key = {
                "scratch": "sensitivity",
                "sharpen": "k",
            }.get(n)
            if n == "denoise":
                value_key = {"gaussian": "sigma", "contraharmonic": "Q",
                             "alpha_trimmed": "d", "k_nearest": "k"}.get(m)
            elif n == "contrast":
                value_key = {"gamma": "gamma", "stretch": "clip"}.get(m)
            value = p.get(value_key, spec[3]) if value_key else spec[3]
            self.sl_main.set_value(value)
            self._param_memory[(m, self._ksize())] = self.sl_main.value()

        for b in blockers:
            b.unblock()
        self._apply_enabled_look()
