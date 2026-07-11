#!/usr/bin/env bash
set -euo pipefail

# Colab 第二阶段：完整公开数据预处理。
# 前提：第一阶段 smoke 已经无报错。
# 这个脚本会断点续跑：已完成的 MEAD 身份和 CREMA-D 文件会跳过。
#
# 用法：
#   bash scripts/colab_run_stage2_full_preprocess.sh

REPRO_ROOT="${REPRO_ROOT:-/content/cmet-repro-colab}"
CMET_ROOT="${CMET_ROOT:-/content/C-MET}"
MEAD_SHARED_ROOT="${MEAD_SHARED_ROOT:-/content/drive/MyDrive/MEAD}"
OUT_ROOT="${OUT_ROOT:-/content/drive/MyDrive/C-MET-full}"
WORK_ROOT="${WORK_ROOT:-/content/cmet_public_data}"
REPORT_ROOT="${REPORT_ROOT:-${OUT_ROOT}/reports}"
MEAD_OUT="${MEAD_OUT:-${OUT_ROOT}/dataset/MEAD/FPS25}"
CREMAD_OUT="${CREMAD_OUT:-${OUT_ROOT}/dataset/CREMA_D/FPS25}"
CMET_COMMIT="${CMET_COMMIT:-0ca437cf7a8129c6a5dca1e2667a588410822bbe}"

echo "== C-MET 复现：第二阶段完整数据预处理 =="
echo "输出目录: ${OUT_ROOT}"

if [[ ! -d "/content/drive/MyDrive" ]]; then
  echo "没有检测到 /content/drive/MyDrive。请先在 Notebook 里运行 drive.mount('/content/drive')。"
  exit 1
fi

if [[ ! -d "${REPRO_ROOT}/.git" ]]; then
  echo "没有找到复现仓库：${REPRO_ROOT}"
  echo "请先运行第一阶段，或克隆：git clone https://github.com/ChengyangHe-ux/cmet-repro-colab.git ${REPRO_ROOT}"
  exit 1
fi

cd "${REPRO_ROOT}"
git pull --ff-only

if [[ ! -d "${CMET_ROOT}/.git" ]]; then
  git clone https://github.com/ChanHyeok-Choi/C-MET.git "${CMET_ROOT}"
fi
git -C "${CMET_ROOT}" fetch origin "${CMET_COMMIT}"
git -C "${CMET_ROOT}" switch --detach "${CMET_COMMIT}"

if [[ ! -d "${MEAD_SHARED_ROOT}" ]]; then
  echo "没有找到 MEAD 官方数据聚合目录：${MEAD_SHARED_ROOT}"
  echo "请把所有可访问的官方视频 Part 快捷方式聚合到该目录，然后先重跑第一阶段。"
  exit 1
fi

mkdir -p "${MEAD_OUT}" "${CREMAD_OUT}" "${WORK_ROOT}" "${REPORT_ROOT}"

echo "== 1/2 MEAD 完整预处理：47 个官方身份，逐身份断点续跑 =="
python "${REPRO_ROOT}/scripts/prepare_public_datasets.py" mead \
  --shared-root "${MEAD_SHARED_ROOT}" \
  --cmet-root "${CMET_ROOT}" \
  --out-root "${MEAD_OUT}" \
  --work-root "${WORK_ROOT}" \
  --report-root "${REPORT_ROOT}"

echo "== 2/2 CREMA-D 完整预处理：只下载 benchmark 实际引用文件 =="
python "${REPRO_ROOT}/scripts/prepare_public_datasets.py" crema-d \
  --cmet-root "${CMET_ROOT}" \
  --out-root "${CREMAD_OUT}" \
  --work-root "${WORK_ROOT}" \
  --report-root "${REPORT_ROOT}" \
  --cleanup-source

echo "== 完整数据预处理完成 =="
echo "报告目录：${REPORT_ROOT}"
