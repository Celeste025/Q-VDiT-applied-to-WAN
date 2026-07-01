#!/usr/bin/env bash
set -eo pipefail

# Move sample_*.mp4 from legacy *_opensora_opensora/ into *_videos_opensora/.
# Usage: bash scripts/fix_vbench_raw_dirs.sh [tag ...]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
TAGS=("$@")
if [[ ${#TAGS[@]} -eq 0 ]]; then
  TAGS=(w4a6 w4a4 w8a8 w3a6)
fi

for TAG in "${TAGS[@]}"; do
  LEGACY="${EVAL_DIR}/${TAG}_videos_opensora_opensora"
  CANON="${EVAL_DIR}/${TAG}_videos_opensora"
  if [[ ! -d "${LEGACY}" ]]; then
    echo "[skip] ${TAG}: no legacy dir ${LEGACY}"
    continue
  fi
  n_legacy="$(find "${LEGACY}" -maxdepth 1 -name 'sample_*.mp4' | wc -l)"
  if [[ "${n_legacy}" -eq 0 ]]; then
    echo "[skip] ${TAG}: empty legacy dir"
    continue
  fi
  mkdir -p "${CANON}"
  echo "[fix] ${TAG}: move ${n_legacy} files ${LEGACY} -> ${CANON}"
  mv "${LEGACY}"/sample_*.mp4 "${CANON}/"
  rmdir "${LEGACY}" 2>/dev/null || true
  echo "[ok] ${TAG}: canonical raw dir ${CANON} ($(find "${CANON}" -name 'sample_*.mp4' | wc -l) samples)"
done
