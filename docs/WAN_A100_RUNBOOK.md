# Wan2.1 Q-VDiT 量化实验 — A100 服务器运行手册

本文档供 A100 服务器工程师直接按步骤部署环境并运行 **Wan2.1-T2V-1.3B** 量化实验。  
更通用的双环境说明见 [`ENV_SETUP.md`](./ENV_SETUP.md)。

---

## 实验目标

在 A100（40GB / 80GB）上完成 Wan PTQ 全流程：

1. 标定数据收集（calib data）
2. delta block-reconstruction 标定（calib）
3. 量化推理（quant inference）

**Full 分辨率**：480×832×81，50 steps，bfloat16。

---

## 执行优先级（必读）

1. **先找空闲 GPU**，确认无他人任务占用后再启动（见 [§6 GPU 选择](#6-gpu-选择)）。
2. **四组量化实验（w8a8 / w4a6 / w4a4 / w3a6）先只跑 w8a8**。  
   w8a8 全流程跑通、产出 `ckpt.pth` 和 smoke 视频后，再依次跑 w4a6 → w4a4 → w3a6。
3. **务必使用 A100 专用脚本** `scripts/run_wan_pipeline_a100.sh`，不要用 `run_wan_pipeline.sh`（4090 非 checkpoint 版）。

---

## 1. 硬件与软件前提

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA A100 40GB 或 80GB |
| CUDA | 12.x |
| 系统工具 | conda、git、nvidia-driver 已安装 |
| 并行 | 单卡可跑 full pipeline；多实验占不同 GPU |

---

## 2. 代码与依赖仓库

```bash
# 主仓库
cd /path/to/Q-VDiT
git checkout main
git pull

# VBench 评测需要 sibling 仓库（与 Q-VDiT 同级）
cd /path/to/parent
git clone https://github.com/wlfeng0509/ViDiT-Q-viditq.git
```

期望目录结构：

```text
parent/
├── Q-VDiT/
└── ViDiT-Q-viditq/
    └── eval/video/Vbench/
```

---

## 3. Wan2.1 权重

需要 **Diffusers 格式**目录，含 `transformer/`、`vae/`、`text_encoder/` 等：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir /path/to/Wan2.1-T2V-1.3B-Diffusers
```

或从已有机器 `rsync` / `scp` 拷贝。

写入环境变量（建议加入 `~/.bashrc`）：

```bash
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/path/to/Q-VDiT
```

---

## 4. Conda 环境

OpenSora 与 Wan **依赖不兼容**，必须使用两个独立 conda 环境，**不可混装**。

### 4.1 `wan-qvdit`（Wan 量化 / 推理，必装）

```bash
cd /path/to/Q-VDiT
bash scripts/setup_env_wan.sh
conda activate wan-qvdit

python -c "from diffusers import WanPipeline; import qdiff; print('wan-qvdit OK')"
python -c "
import json, os
from pathlib import Path
p = Path(os.environ['WAN_MODEL_PATH'])
cfg = json.loads((p / 'transformer/config.json').read_text())
assert cfg['num_layers'] == 30
print('Wan model OK:', p)
"
```

| 包 | 版本 |
|----|------|
| Python | 3.10 |
| torch | cu121（pip 官方 wheel） |
| diffusers | **>= 0.33.0** |
| transformers | >= 4.40.0 |

**禁止**在此环境安装 opensora 或 diffusers 0.24。

### 4.2 `qvdit`（VBench 评测，可选）

仅在做 VBench 子集打分时需要：

```bash
bash scripts/setup_env.sh
conda activate qvdit
python -c "import qdiff, opensora; print('qvdit OK')"
```

Wan 视频在 `wan-qvdit` 生成；VBench 在 `qvdit` 中运行。

---

## 5. A100 配置说明

| 项目 | A100 full | 4090 smoke（勿用于 A100 full） |
|------|-----------|--------------------------------|
| 入口脚本 | `scripts/run_wan_pipeline_a100.sh` | `scripts/run_wan_smoke.sh` |
| 量化配置 | `wan/configs/quant/a100/*.yaml` | `wan/configs/quant/*_smoke.yaml` |
| 推理配置 | `wan/configs/infer.yaml` | `wan/configs/infer_smoke.yaml` |
| 分辨率 | **480×832×81** | 256×448×17 |
| `grad_checkpoint` | **True** | False |
| delta 优化 | 开启 | smoke 关闭 |
| `n_spatial_token` | **1792** | 3584 |
| 输出目录 | `logs_wan/{exp}_a100/` | `logs_wan/{exp}_smoke/` |

**推理参数**（`wan/configs/infer.yaml`，Wan2.1 native 480P 默认）：

| 参数 | 值 |
|------|-----|
| height × width × frames | 480 × 832 × 81 |
| num_inference_steps | 50 |
| guidance_scale | 6.0 |
| flow_shift | 8.0 |
| dtype | bfloat16 |

**显存优化**（标定阶段建议开启）：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 6. GPU 选择

启动实验前**必须确认 GPU 空闲**，避免与他人任务冲突：

```bash
nvidia-smi
# 或持续观察
watch -n 2 nvidia-smi
```

选择 **显存占用接近 0、GPU-Util 为 0%** 的卡。示例：若 GPU 2 空闲，则：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_wan_pipeline_a100.sh w8a8_ours 2
```

> 脚本第二个参数 `gpu_id` 与 `CUDA_VISIBLE_DEVICES` 应保持一致（脚本内部以 `gpu 0` 映射到可见设备中的第一张卡）。

---

## 7. Smoke 验证（可选，约 10 分钟）

确认 GPU、权重、代码链路正常（**非 full 分辨率**）：

```bash
conda activate wan-qvdit
export PYTHONPATH=/path/to/Q-VDiT
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers

CUDA_VISIBLE_DEVICES=<空闲GPU> bash scripts/run_wan_smoke.sh <空闲GPU> w8a8_ours_smoke
```

成功标志：

- `logs_wan/w8a8_ours_smoke/ckpt.pth` 存在
- `logs_wan/w8a8_ours_smoke/` 下有 smoke 视频

---

## 8. A100 Full 量化实验（主流程）

### 8.1 第一步：只跑 w8a8

```bash
conda activate wan-qvdit
export PYTHONPATH=/path/to/Q-VDiT
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 替换 <GPU> 为 nvidia-smi 看到的空闲卡号
CUDA_VISIBLE_DEVICES=<GPU> bash scripts/run_wan_pipeline_a100.sh w8a8_ours <GPU> \
  2>&1 | tee logs_wan/w8a8_a100_pipeline.log
```

建议用 **tmux** 防止 SSH 断开：

```bash
tmux new -s wan_w8a8
# 在 tmux 内执行上述命令
```

**三阶段说明**（脚本自动串联）：

| 阶段 | 说明 | 共享/输出 |
|------|------|-----------|
| [1/3] calib data | 10 条 prompt，480×832×81 | `logs_wan/calib_data/calib_data.pt`（各实验共享） |
| [2/3] calib | delta block-recon + grad checkpoint | `logs_wan/w8a8_ours_a100/ckpt.pth` |
| [3/3] quant inference | 10 条 smoke 视频 | `logs_wan/w8a8_ours_a100/*.mp4` |

**w8a8 成功标志**：

- [ ] `logs_wan/calib_data/calib_data.pt` 存在
- [ ] `logs_wan/w8a8_ours_a100/ckpt.pth` 存在
- [ ] `logs_wan/w8a8_ours_a100/` 下有 10 个 `.mp4`

w8a8 标定约需数小时，请耐心等待日志出现 `=== w8a8_ours A100 DONE ===`。

### 8.2 后续：w4a6 / w4a4 / w3a6（w8a8 通过后再跑）

`calib_data.pt` 已存在时，后续实验会复用，无需重跑 [1/3]（脚本仍会调用 get_calib_data，可覆盖或跳过视实现而定）。

```bash
# w8a8 全部通过后，按需依次启动（各占一卡可并行）
CUDA_VISIBLE_DEVICES=<GPU0> bash scripts/run_wan_pipeline_a100.sh w4a6_ours <GPU0>
CUDA_VISIBLE_DEVICES=<GPU1> bash scripts/run_wan_pipeline_a100.sh w4a4_ours <GPU1>
CUDA_VISIBLE_DEVICES=<GPU2> bash scripts/run_wan_pipeline_a100.sh w3a6_ours <GPU2>
```

| exp_name | 精度 | 输出目录 | 标定 iters（约） |
|----------|------|----------|------------------|
| w8a8_ours | W8A8 | `logs_wan/w8a8_ours_a100/` | 5000 |
| w4a6_ours | W4A6 mixed | `logs_wan/w4a6_ours_a100/` | 10000 |
| w4a4_ours | W4A4 | `logs_wan/w4a4_ours_a100/` | 10000 |
| w3a6_ours | W3A6 | `logs_wan/w3a6_ours_a100/` | 15000 |

---

## 9. VBench 子集-25 评测（可选）

### 9.1 生成视频（`wan-qvdit`）

```bash
conda activate wan-qvdit
export PYTHONPATH=/path/to/Q-VDiT
EVAL=logs_wan/eval_vbench_subset25
mkdir -p "$EVAL/fp16_videos" "$EVAL/w8a8_videos"

# FP16 baseline（25 条）
CUDA_VISIBLE_DEVICES=<GPU> python wan/scripts/fp_vbench_generate.py \
  --model-path "$WAN_MODEL_PATH" \
  --save-dir "$EVAL/fp16_videos" \
  --infer-config wan/configs/infer.yaml

# W8A8（需 w8a8_a100 标定完成）
CUDA_VISIBLE_DEVICES=<GPU> python wan/scripts/vbench_generate.py \
  --model-path "$WAN_MODEL_PATH" \
  --outdir "$EVAL/w8a8_videos" \
  --quant-config wan/configs/quant/a100/w8a8_ours.yaml \
  --quant-ckpt logs_wan/w8a8_ours_a100/ckpt.pth \
  --infer-config wan/configs/infer.yaml \
  --part-fp
```

> **注意**：A100 请使用 `wan/configs/quant/a100/` 下的配置和 `logs_wan/*_a100/` 下的 ckpt，  
> 不要使用 `scripts/run_wan_vbench.sh` 里默认的非 A100 路径。

### 9.2 跑 VBench 打分（`qvdit`）

```bash
conda activate qvdit
export PYTHONPATH=/path/to/Q-VDiT

bash scripts/eval_vbench_wan_subset25.sh \
  logs_wan/eval_vbench_subset25/fp16_videos \
  logs_wan/eval_vbench_subset25/vbench_results_fp16 \
  <GPU>
```

---

## 10. 常见问题

| 现象 | 处理 |
|------|------|
| calib 阶段 OOM | 确认使用 `run_wan_pipeline_a100.sh` + `wan/configs/quant/a100/*.yaml`（`grad_checkpoint: True`） |
| `WanPipeline` import 失败 | 检查是否在 `wan-qvdit` 环境，diffusers >= 0.33 |
| 模型路径错误 | 检查 `WAN_MODEL_PATH/transformer/config.json`：`num_layers=30`, `text_dim=4096` |
| VBench 找不到 | 确认 `../ViDiT-Q-viditq/eval/video/Vbench` 与 Q-VDiT 同级 |
| git push 失败 | 多为网络问题，重试或配置代理 |

---

## 11. 禁止事项

- 不要在 `wan-qvdit` 中安装 / 升级 diffusers 到 0.24
- 不要在标定进行中 `pip upgrade` 任何包
- 不要用 `run_wan_pipeline.sh` 跑 A100 full 480P
- 不要未确认 GPU 空闲就占卡跑实验
- **不要跳过 w8a8 直接跑 w4a6 / w4a4 / w3a6**

---

## 12. 关键路径速查

| 用途 | 路径 |
|------|------|
| A100 入口脚本 | `scripts/run_wan_pipeline_a100.sh` |
| A100 量化配置 | `wan/configs/quant/a100/` |
| Full 推理配置 | `wan/configs/infer.yaml` |
| 标定数据 | `logs_wan/calib_data/calib_data.pt` |
| w8a8 输出 | `logs_wan/w8a8_ours_a100/` |
| Wan 环境安装 | `scripts/setup_env_wan.sh` |
| 环境总览 | `docs/ENV_SETUP.md` |

---

## 13. 完成后反馈

w8a8 跑通后，请反馈：

1. `nvidia-smi` 使用的 GPU 编号与型号（40G / 80G）
2. `logs_wan/w8a8_ours_a100/ckpt.pth` 是否生成
3. pipeline 总耗时与是否有 OOM / 报错
4. smoke 视频路径（便于定性检查）

便于决定是否继续 w4a6 / w4a4 / w3a6 及是否需要调 batch / 显存参数。
