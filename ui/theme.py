"""
Module ui/theme.py
Bảng màu và stylesheet dùng chung cho toàn bộ giao diện.
"""
import os

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets").replace("\\", "/")

ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
BG = "#f3f5f9"
CARD = "#ffffff"
BORDER = "#e2e8f0"
TEXT = "#0f172a"
MUTED = "#64748b"
OK = "#059669"
WARN = "#d97706"
ERR = "#dc2626"

APP_STYLESHEET = f"""
QMainWindow, QWidget#root {{
    background: {BG};
}}
QWidget {{
    font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
    font-size: 13px;
    color: {TEXT};
}}
QToolTip {{
    background: {TEXT};
    color: white;
    border: none;
    padding: 6px 8px;
    border-radius: 4px;
}}

/* ---- Thanh tiêu đề ---- */
QFrame#header {{
    background: {CARD};
    border-bottom: 1px solid {BORDER};
}}
QLabel#appTitle {{
    font-size: 17px;
    font-weight: 700;
}}
QLabel#appSubtitle {{
    color: {MUTED};
    font-size: 12px;
}}

/* ---- Nút ---- */
QPushButton, QToolButton {{
    background: {CARD};
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
    color: #1e293b;
}}
QPushButton:hover, QToolButton:hover {{
    background: #f1f5f9;
    border-color: #94a3b8;
}}
QPushButton:disabled, QToolButton:disabled {{
    color: #94a3b8;
    background: #f8fafc;
}}
QToolButton::menu-indicator {{
    subcontrol-position: right center;
    right: 6px;
}}
QToolButton {{
    padding-right: 22px;
}}
QPushButton#primary {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: white;
}}
QPushButton#primary:hover {{
    background: {ACCENT_DARK};
}}
QPushButton#primary:disabled {{
    background: #93c5fd;
    border-color: #93c5fd;
}}
QPushButton#autoButton {{
    background: {TEXT};
    border: 1px solid {TEXT};
    color: white;
    font-size: 14px;
    padding: 11px 14px;
    border-radius: 9px;
}}
QPushButton#autoButton:hover {{
    background: #1e293b;
}}
QPushButton#ghost {{
    background: transparent;
    border: 1px solid transparent;
    color: {MUTED};
    padding: 6px 10px;
}}
QPushButton#ghost:hover {{
    background: #e2e8f0;
    color: {TEXT};
}}

/* ---- Thẻ ---- */
QFrame#card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QLabel#cardTitle {{
    font-weight: 700;
    font-size: 13px;
}}
QLabel#muted {{
    color: {MUTED};
    font-size: 12px;
}}
QLabel#formula {{
    color: #475569;
    font-size: 12px;
    font-family: 'Cambria Math', 'Cambria', 'Times New Roman', serif;
    background: #f8fafc;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
}}
QLabel#sectionTitle {{
    color: {MUTED};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel#stepBadge {{
    background: #e0e7ff;
    color: #3730a3;
    border-radius: 10px;
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
    font-weight: 700;
    font-size: 11px;
    qproperty-alignment: AlignCenter;
}}
QLabel#stepBadgeOff {{
    background: #f1f5f9;
    color: #94a3b8;
    border-radius: 10px;
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
    font-weight: 700;
    font-size: 11px;
    qproperty-alignment: AlignCenter;
}}
QLabel#imageView {{
    background: #eef2f7;
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    color: #94a3b8;
}}
QLabel#report {{
    background: #f8fafc;
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 10px;
    color: #334155;
    font-size: 12px;
}}

/* ---- Điều khiển ---- */
QCheckBox {{
    spacing: 8px;
    font-weight: 700;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 2px solid #94a3b8;
    background: white;
}}
QCheckBox::indicator:checked {{
    border: 2px solid {ACCENT};
    background: {ACCENT};
    image: url({_ASSETS}/check.svg);
}}
QComboBox {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 8px;
    min-height: 18px;
}}
QComboBox:hover {{
    border-color: #94a3b8;
}}
QComboBox:disabled {{
    color: #94a3b8;
}}
QComboBox QAbstractItemView {{
    background: white;
    border: 1px solid {BORDER};
    selection-background-color: #dbeafe;
    selection-color: {TEXT};
    outline: 0;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal:disabled {{
    background: #cbd5e1;
}}
QSlider::handle:horizontal {{
    background: white;
    border: 2px solid {ACCENT};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:disabled {{
    border-color: #cbd5e1;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QSplitter::handle {{
    background: transparent;
}}
QStatusBar {{
    background: {CARD};
    border-top: 1px solid {BORDER};
    color: {MUTED};
    font-size: 12px;
}}
QMenu {{
    background: white;
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 7px 18px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: #dbeafe;
    color: {TEXT};
}}
"""
