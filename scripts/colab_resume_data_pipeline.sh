#!/usr/bin/env bash
set -euo pipefail

# C-MET Colab 数据流水线续跑控制器。
#
# 常用方式：
#   bash scripts/colab_resume_data_pipeline.sh
#   CMET_CONTINUE_FULL=1 bash scripts/colab_resume_data_pipeline.sh
#   bash scripts/colab_resume_data_pipeline.sh status
#   bash scripts/colab_resume_data_pipeline.sh smoke
#   bash scripts/colab_resume_data_pipeline.sh full

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="${REPRO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-/content/drive/MyDrive/C-MET-full}"
REPORT_ROOT="${REPORT_ROOT:-${OUT_ROOT}/reports}"
MODE="${1:-auto}"
CONTINUE_FULL="${CMET_CONTINUE_FULL:-0}"
STATUS_SCRIPT="${REPRO_ROOT}/scripts/data_pipeline_status.py"
STAGE1_SCRIPT="${REPRO_ROOT}/scripts/colab_run_stage1_data_smoke.sh"
STAGE2_SCRIPT="${REPRO_ROOT}/scripts/colab_run_stage2_full_preprocess.sh"

usage() {
  cat <<'EOF'
用法：bash scripts/colab_resume_data_pipeline.sh [auto|status|smoke|full]

  auto    根据状态自动选择下一阶段；默认完成 smoke 后暂停检查报告
  status  只打印当前进度
  smoke   强制执行来源预检、MEAD smoke 和 CREMA-D smoke
  full    强制执行 MEAD/CREMA-D 完整预处理

设置 CMET_CONTINUE_FULL=1 后，auto 会在 smoke 成功后继续完整预处理。
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "缺少脚本：${path}" >&2
    exit 1
  fi
}

show_status() {
  python3 "${STATUS_SCRIPT}" \
    --out-root "${OUT_ROOT}" \
    --report-root "${REPORT_ROOT}"
}

next_stage() {
  python3 "${STATUS_SCRIPT}" \
    --out-root "${OUT_ROOT}" \
    --report-root "${REPORT_ROOT}" \
    --field next_stage
}

print_feature_handoff() {
  cat <<EOF

数据预处理阶段已经完成。
下一步请打开 notebooks/C-MET_Full_Reproduction_Colab.ipynb，按顺序运行：
  1. emotion2vec+large 完整特征抽取
  2. EDTalk 完整特征抽取
  3. 完整数据合同校验
  4. 构建训练缓存
  5. 2 step 训练 smoke

状态报告目录：${REPORT_ROOT}
EOF
}

for required in "${STATUS_SCRIPT}" "${STAGE1_SCRIPT}" "${STAGE2_SCRIPT}"; do
  require_file "${required}"
done

case "${MODE}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  status)
    show_status
    exit 0
    ;;
  smoke)
    bash "${STAGE1_SCRIPT}"
    show_status
    exit 0
    ;;
  full)
    bash "${STAGE2_SCRIPT}"
    show_status
    if [[ "$(next_stage)" == "features" ]]; then
      print_feature_handoff
    fi
    exit 0
    ;;
  auto)
    ;;
  *)
    echo "未知模式：${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

stage="$(next_stage)"
echo "== 当前自动判定阶段：${stage} =="
show_status

case "${stage}" in
  features)
    print_feature_handoff
    ;;
  full)
    echo "smoke 已通过，继续完整数据预处理。"
    bash "${STAGE2_SCRIPT}"
    show_status
    if [[ "$(next_stage)" == "features" ]]; then
      print_feature_handoff
    fi
    ;;
  smoke)
    echo "开始来源预检和小样本 smoke。"
    bash "${STAGE1_SCRIPT}"
    show_status
    stage="$(next_stage)"
    if [[ "${stage}" == "full" && "${CONTINUE_FULL}" == "1" ]]; then
      echo "smoke 已通过，CMET_CONTINUE_FULL=1，继续完整数据预处理。"
      bash "${STAGE2_SCRIPT}"
      show_status
      if [[ "$(next_stage)" == "features" ]]; then
        print_feature_handoff
      fi
    elif [[ "${stage}" == "full" ]]; then
      cat <<EOF

smoke 已通过。请先检查 ${REPORT_ROOT} 中的报告。
确认无误后，重新运行同一条命令即可进入完整预处理：
  bash scripts/colab_resume_data_pipeline.sh

也可以一次连续执行：
  CMET_CONTINUE_FULL=1 bash scripts/colab_resume_data_pipeline.sh
EOF
    fi
    ;;
  *)
    echo "无法识别下一阶段：${stage}" >&2
    exit 1
    ;;
esac
