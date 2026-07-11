#!/usr/bin/env bash
set -euo pipefail

# Colab 第一阶段：公开数据来源预检 + 小样本 smoke test。
# 作用：
# 1. 更新复现仓库到 GitHub 最新版本；
# 2. 确认官方 C-MET 仓库存在并固定到论文代码版本；
# 3. 检查 MyDrive/MEAD 快捷方式；
# 4. 跑 MEAD 来源预检、MEAD smoke、CREMA-D smoke。
#
# 用法：
#   bash scripts/colab_run_stage1_data_smoke.sh

REPRO_ROOT="${REPRO_ROOT:-/content/cmet-repro-colab}"
CMET_ROOT="${CMET_ROOT:-/content/C-MET}"
MEAD_SHARED_ROOT="${MEAD_SHARED_ROOT:-/content/drive/MyDrive/MEAD}"
OUT_ROOT="${OUT_ROOT:-/content/drive/MyDrive/C-MET-full}"
WORK_ROOT="${WORK_ROOT:-/content/cmet_public_data}"
REPORT_ROOT="${REPORT_ROOT:-${OUT_ROOT}/reports}"
MEAD_OUT="${MEAD_OUT:-${OUT_ROOT}/dataset/MEAD/FPS25}"
CREMAD_OUT="${CREMAD_OUT:-${OUT_ROOT}/dataset/CREMA_D/FPS25}"
CMET_COMMIT="${CMET_COMMIT:-0ca437cf7a8129c6a5dca1e2667a588410822bbe}"

echo "== C-MET 复现：第一阶段数据 smoke =="
echo "复现仓库: ${REPRO_ROOT}"
echo "官方 C-MET: ${CMET_ROOT}"
echo "MEAD 快捷方式: ${MEAD_SHARED_ROOT}"
echo "输出目录: ${OUT_ROOT}"

if [[ ! -d "/content/drive/MyDrive" ]]; then
  echo "没有检测到 /content/drive/MyDrive。请先在 Notebook 里运行 drive.mount('/content/drive')。"
  exit 1
fi

if [[ ! -d "${REPRO_ROOT}/.git" ]]; then
  echo "没有找到复现仓库：${REPRO_ROOT}"
  echo "请先在 Colab 终端执行："
  echo "git clone https://github.com/ChengyangHe-ux/cmet-repro-colab.git ${REPRO_ROOT}"
  exit 1
fi

cd "${REPRO_ROOT}"
echo "== 更新复现仓库 =="
git pull --ff-only

echo "== 准备官方 C-MET 仓库 =="
if [[ ! -d "${CMET_ROOT}/.git" ]]; then
  git clone https://github.com/ChanHyeok-Choi/C-MET.git "${CMET_ROOT}"
fi
git -C "${CMET_ROOT}" fetch origin "${CMET_COMMIT}"
git -C "${CMET_ROOT}" switch --detach "${CMET_COMMIT}"

if [[ ! -d "${MEAD_SHARED_ROOT}" ]]; then
  echo "没有找到 MEAD 官方快捷方式：${MEAD_SHARED_ROOT}"
  echo "请把 MEAD 官方 Part0 文件夹添加到 MyDrive 根目录，并命名为 MEAD。"
  exit 1
fi

mkdir -p "${MEAD_OUT}" "${CREMAD_OUT}" "${WORK_ROOT}" "${REPORT_ROOT}"

echo "== 1/3 MEAD 来源预检：只检查 video.tar / video_*.tar，不复制数据 =="
python "${REPRO_ROOT}/scripts/prepare_public_datasets.py" mead \
  --shared-root "${MEAD_SHARED_ROOT}" \
  --cmet-root "${CMET_ROOT}" \
  --out-root "${MEAD_OUT}" \
  --work-root "${WORK_ROOT}" \
  --report-root "${REPORT_ROOT}" \
  --check-only

echo "== 2/3 MEAD smoke：只处理 1 个身份、2 条视频 =="
python "${REPRO_ROOT}/scripts/prepare_public_datasets.py" mead \
  --shared-root "${MEAD_SHARED_ROOT}" \
  --cmet-root "${CMET_ROOT}" \
  --out-root "${MEAD_OUT}" \
  --work-root "${WORK_ROOT}" \
  --report-root "${REPORT_ROOT}" \
  --limit-identities 1 \
  --limit-videos 2 \
  --keep-work

echo "== 3/3 CREMA-D smoke：只处理 2 条 benchmark 视频 =="
python "${REPRO_ROOT}/scripts/prepare_public_datasets.py" crema-d \
  --cmet-root "${CMET_ROOT}" \
  --out-root "${CREMAD_OUT}" \
  --work-root "${WORK_ROOT}" \
  --report-root "${REPORT_ROOT}" \
  --limit-videos 2

echo "== 第一阶段完成 =="
echo "请检查报告目录：${REPORT_ROOT}"
echo "如果没有报错，下一步再运行完整预处理。"
