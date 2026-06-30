EXP_NAME='w8a8_ours_smoke'

CFG="./t2v/configs/quant/opensora/16x512x512.py"
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"
OUTDIR="./logs_smoke/$EXP_NAME"
GPU_ID="${1:-0}"

CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/quant_txt2video.py $CFG \
    --outdir $OUTDIR --ckpt_path $CKPT_PATH \
    --dataset_type opensora \
    --part_fp \
    --precompute_text_embeds ./t2v/utils_files/text_embeds.pth \
    --prompt_path t2v/assets/texts/t2v_samples_10.txt \
    --num_videos 2
