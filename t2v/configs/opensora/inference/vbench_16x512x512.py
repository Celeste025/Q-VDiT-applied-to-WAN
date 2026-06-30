num_frames = 16
fps = 24 // 3
image_size = (512, 512)

model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    enable_flashattn=False,
    enable_layernorm_kernel=False,
    from_pretrained="PRETRAINED_MODEL",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/data/home/jinqiwen/workspace/video-distilation/Q-VDiT/logs/vae_ckpt",
    micro_batch_size=128,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/data/home/jinqiwen/workspace/video-distilation/Q-VDiT/logs",
    local_cache=True,
    save_pretrained="/data/home/jinqiwen/workspace/video-distilation/Q-VDiT/logs/t5-v1_1-xxl",
    model_max_length=120,
)
precompute_text_embeds = "/data/home/jinqiwen/workspace/video-distilation/Q-VDiT/logs/eval_vbench_subset25/text_embeds.pth"
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=4.0,
)
dtype = "fp16"

batch_size = 1
seed = 42
prompt_path = "/data/home/jinqiwen/workspace/video-distilation/Q-VDiT/logs/eval_vbench_subset25/prompts.txt"
