# -*- coding: utf-8 -*-
"""
main.py —— 项目统一运行入口

依次执行：
    1. 01_data_exploration.py  数据读取、清洗与探索性分析
    2. 02_modeling.py         特征工程、模型训练与正则化对比

运行方式：
    python code/main.py
所有控制台输出同时写入 output/console_log.txt。
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS = ["01_data_exploration.py", "02_modeling.py"]


def main() -> None:
    log_lines = []
    for step in STEPS:
        print(f"\n===== 运行 {step} =====")
        result = subprocess.run(
            [sys.executable, str(CODE_DIR / step)],
            capture_output=True, text=True, encoding="utf-8", errors="strict",
        )
        log_lines.append(f"===== {step} =====\n{result.stdout}\n{result.stderr}\n")
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise SystemExit(f"{step} 运行失败，退出码 {result.returncode}")

    (OUT_DIR / "console_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8")
    print("\n全部步骤运行完成，结果见 output 与 figures 目录。")


if __name__ == "__main__":
    main()
