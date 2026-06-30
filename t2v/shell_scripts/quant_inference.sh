EXP_NAME='w8a8_ours'

CFG="./t2v/configs/quant/opensora/16x512x512.py" # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # your path of splited ckpt
OUTDIR="./logs_100steps/$EXP_NAME"  # your path of the calibration result
GPU_ID=$1

# quant inference (w8a8: no mixed-precision yaml needed)
CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/quant_txt2video.py $CFG \
    --outdir $OUTDIR --ckpt_path $CKPT_PATH  \
    --dataset_type opensora \
    --part_fp \
    --precompute_text_embeds ./t2v/utils_files/text_embeds.pth \
    --prompt_path t2v/assets/texts/t2v_samples_10.txt \
