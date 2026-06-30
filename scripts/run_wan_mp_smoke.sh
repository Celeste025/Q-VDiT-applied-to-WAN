#!/usr/bin/env bash
set -eo pipefail

# L7: mixed-precision Wan quant smoke (w4a6 / w4a4 / w3a6), init-only ckpt gates
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-5}"
EXP="${2:-w4a6_ours_smoke}"
MODEL="${WAN_MODEL_PATH:-${ROOT}/../DVDQuant_rep/pretrained_models/Wan2.1-T2V-1.3B-Diffusers}"
MP_DIR="${ROOT}/wan/configs/mixed_precision"
OUTDIR="${ROOT}/logs_wan/${EXP}"
Q_CFG="${ROOT}/wan/configs/quant/${EXP}.yaml"

MP_CALIB=()
case "${EXP}" in
  w4a6_ours_smoke)
    MP_CALIB=(--time_mp_config_weight "${MP_DIR}/weight_4_mp.yaml" --time_mp_config_act "${MP_DIR}/act_6_mp.yaml")
    ;;
  w3a6_ours_smoke)
    MP_CALIB=(--time_mp_config_weight "${MP_DIR}/weight_3_mp.yaml" --time_mp_config_act "${MP_DIR}/act_6_mp.yaml")
    ;;
  w4a4_ours_smoke)
    MP_CALIB=(--time_mp_config_weight "${MP_DIR}/weight_4_mp.yaml" --time_mp_config_act "${MP_DIR}/act_4_mp.yaml")
    ;;
  *)
    echo "Unknown smoke EXP: ${EXP}" >&2
    exit 1
    ;;
esac

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wan-qvdit
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${ROOT}"
mkdir -p "${OUTDIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/calib.py \
  --calib_config "${Q_CFG}" \
  --calib_data "${ROOT}/logs_wan/calib_data_smoke/calib_data.pt" \
  --outdir "${OUTDIR}" \
  --model-path "${MODEL}" \
  --gpu 0 \
  --part_fp \
  --dtype bfloat16 \
  "${MP_CALIB[@]}"

python -c "from pathlib import Path; from wan.utils.verify import gate_l4; gate_l4(Path('${OUTDIR}/ckpt.pth'))"
echo "=== L7 smoke ${EXP} DONE ==="
