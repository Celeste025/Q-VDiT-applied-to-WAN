EXP_NAME="w8a8_ours_smoke"

CFG="./t2v/configs/quant/opensora/16x512x512.py"
Q_CFG="./t2v/configs/quant/opensora/$EXP_NAME.yaml"
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"
CALIB_DATA_DIR="./logs_100steps/calib_data"
OUTDIR="./logs_smoke/$EXP_NAME"
GPU_ID="${1:-0}"

CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/calib.py $CFG --ckpt_path $CKPT_PATH --calib_config $Q_CFG --outdir $OUTDIR \
    --calib_data $CALIB_DATA_DIR/calib_data.pt \
    --part_fp \
    --precompute_text_embeds ./t2v/utils_files/text_embeds.pth
