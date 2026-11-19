"""
Script chạy toàn bộ kiểm thử đơn vị và nghiệm thu cho restore_app.
"""
import subprocess
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONPATH"] = ROOT_DIR

scripts = [
    "tests/test_core.py",
    "tests/test_ui.py",
    "verify/compare_opencv.py"
]

print("======================================================================")
print("   KIỂM THỬ TOÀN BỘ HỆ THỐNG PHỤC HỒI ẢNH CỔ (INT13146 - PTIT)")
print("======================================================================\n")

all_passed = True
for s in scripts:
    print(f">> Đang chạy: {s} ...")
    ret = subprocess.run([sys.executable, os.path.join(ROOT_DIR, s)], capture_output=True, text=True, env=os.environ)
    if ret.returncode == 0:
        print(ret.stdout.strip())
        print(f">> KẾT QUẢ: PASS\n")
    else:
        print(ret.stderr.strip() or ret.stdout.strip())
        print(f">> KẾT QUẢ: FAILED\n")
        all_passed = False

print("======================================================================")
if all_passed:
    print("   TẤT CẢ BÀI KIỂM THỬ ĐỀU ĐẠT")
else:
    print("   CÓ LỖI XẢY RA TRONG MỘT SỐ BÀI KIỂM THỬ!")
print("======================================================================")
