#!/usr/bin/env python3
"""安装 C-MET 全流程所需依赖，同时保护 Colab 自带 Torch 环境。"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_REQUIREMENTS = Path(__file__).resolve().parents[1] / "configs" / "colab_requirements.txt"


def build_commands(python: str, requirements: Path) -> list[list[str]]:
    return [
        [python, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements)],
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            "face-alignment==1.4.1",
        ],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="安装 C-MET Colab 依赖")
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--python", default=sys.executable, help="要安装依赖的 Python 可执行文件")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行安装")
    args = parser.parse_args()

    requirements = args.requirements.resolve()
    if not requirements.is_file():
        raise FileNotFoundError(f"缺少依赖清单：{requirements}")

    print("不会升级 pip，也不会重新安装 Torch/CUDA。")
    for command in build_commands(args.python, requirements):
        print("$", shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)

    if not args.dry_run:
        print("依赖安装完成。下一步请运行环境自检，不要跳过。")


if __name__ == "__main__":
    main()
