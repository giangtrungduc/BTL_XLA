@echo off
title Khoi Phuc & Nang Cao Chat Luong Anh - BTL XLA PTIT
echo ======================================================================
echo    KHOI PHUC VA NANG CAO CHAT LUONG ANH - BTL XU LY ANH PTIT
echo ======================================================================
echo.

set PYTHONPATH=.

REM --- Kiem tra moi truong ao .venv co ton tai khong ---
if exist ".venv\Scripts\python.exe" (
    echo [OK] Tim thay moi truong ao .venv - dang khoi dong ung dung...
    echo.
    .venv\Scripts\python.exe main.py
) else (
    echo [THONG BAO] Dang khoi dong voi Python he thong...
    python main.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LOI] Co loi xay ra khi khoi chay ung dung!
    echo Vui long kiem tra lai cai dat Python va cac thu vien trong requirements.txt.
    pause
)
