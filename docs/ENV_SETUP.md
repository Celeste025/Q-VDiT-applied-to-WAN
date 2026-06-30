# Q-VDiT 双 Conda 环境说明

本项目同时支持 **OpenSora (STDiT)** 与 **Wan2.1-1.3B** 两条量化管线。两者依赖的 `diffusers` / `torch` 版本**不兼容**，必须使用**两个独立 conda 环境**，不要在同一环境中混装或升级 pip 包。

| 环境名 | 用途 | 安装脚本 | 输出目录 |
|--------|------|----------|----------|
| `qvdit` | OpenSora 16×512×512 PTQ / VBench | `bash scripts/setup_env.sh` | `logs/`, `logs_100steps/` |
| `wan-qvdit` | Wan 480×832×81 PTQ / 推理 | `bash scripts/setup_env_wan.sh` | `logs_wan/` |

VBench 评测脚本 `scripts/eval_vbench_wan_subset25.sh` 在 **`qvdit`** 中运行（需 `clip`、VBench 依赖），Wan 视频生成仍在 **`wan-qvdit`** 中完成。

---

## 1. OpenSora 环境：`qvdit`

**硬件**：RTX 4090 24GB 已验证；多卡 OpenSora calib 可用 GPU 0–3。

**一键安装**：

```bash
cd /path/to/Q-VDiT
bash scripts/setup_env.sh          # 默认 env 名: qvdit
conda activate qvdit
export PYTHONPATH=/path/to/Q-VDiT
```

**核心版本（由 `setup_env.sh` 固定）**：

| 包 | 版本 |
|----|------|
| Python | 3.10 |
| torch / torchvision / torchaudio | 2.1.1 / 0.16.1 / 2.1.1 (cu121) |
| diffusers | **0.24.0** |
| transformers | 4.36.2 |
| huggingface_hub | **0.23.5**（与 diffusers 0.24 匹配，勿升级） |
| xformers | 0.0.23 |
| colossalai | 0.4.0 |
| mmengine | latest from pip |

**Editable 安装**：

```bash
pip install -e .          # qdiff
pip install -e ./t2v      # opensora
```

**OpenSora 权重**：需自行下载并 `split_ckpt`，见根目录 `README.md`。

**典型命令**：

```bash
conda activate qvdit
export PYTHONPATH=/path/to/Q-VDiT
CUDA_VISIBLE_DEVICES=0 bash scripts/run_quant_pipeline.sh w8a8_ours 0
```

---

## 2. Wan 环境：`wan-qvdit`

**硬件**：

- **RTX 4090 24GB**：仅 smoke（256×448×17）+ init-only 标定已通过；full 480×832 + delta block-recon 易 OOM。
- **A100 40GB/80GB**：推荐跑 full 管线，使用 `scripts/run_wan_pipeline_a100.sh` 与 `wan/configs/quant/a100/` 配置。

**一键安装**：

```bash
cd /path/to/Q-VDiT
bash scripts/setup_env_wan.sh
conda activate wan-qvdit
export PYTHONPATH=/path/to/Q-VDiT
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export HF_ENDPOINT=https://hf-mirror.com   # 可选，国内镜像
```

**核心版本（由 `setup_env_wan.sh` 固定）**：

| 包 | 版本 |
|----|------|
| Python | 3.10 |
| torch | 2.x cu121（pip 官方 wheel） |
| diffusers | **>= 0.33.0**（WanPipeline） |
| transformers | >= 4.40.0 |
| accelerate, omegaconf, safetensors, einops, decord | pip latest |

**Editable 安装**：

```bash
pip install -e .          # qdiff（含 model_type=wan）
# 不要 pip install -e ./t2v
```

**Wan 权重**：Diffusers 格式目录，需含 `transformer/`、`vae/`、`text_encoder/` 等。默认路径：

```text
../DVDQuant_rep/pretrained_models/Wan2.1-T2V-1.3B-Diffusers
```

或通过环境变量 `WAN_MODEL_PATH` 指定。

**典型命令（4090 smoke）**：

```bash
conda activate wan-qvdit
CUDA_VISIBLE_DEVICES=4 bash scripts/run_wan_smoke.sh 4
```

**典型命令（A100 full）**：

```bash
conda activate wan-qvdit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wan_pipeline_a100.sh w8a8_ours 0
```

---

## 3. 并行运行规范（多卡服务器）

| 任务 | 环境 | 建议 GPU | 输出 |
|------|------|----------|------|
| OpenSora calib / VBench | `qvdit` | 0–3 | `logs_100steps/` |
| Wan 实验 | `wan-qvdit` | 4+ 或 A100 独占 | `logs_wan/` |

**禁止**：在 OpenSora 标定进程运行期间，对 `qvdit` 执行 `pip upgrade`（会破坏 diffusers/huggingface_hub 组合）。

---

## 4. A100 与 4090 配置差异

| 项目 | 4090 smoke | A100 full (`wan/configs/quant/a100/`) |
|------|------------|----------------------------------------|
| 分辨率 | 256×448×17 | 480×832×81 |
| delta 优化 | 关闭（`params: null`） | **开启**（5000 iters） |
| grad checkpoint | 关闭 | **开启**（block-recon 必备） |
| calib batch | 1 | 1（可改为 2 试验） |

480×720：可复制 `wan/configs/infer.yaml` 改 `width: 720`，并依 latent 重算 `n_spatial_token`（见 `w8a8_ours.yaml` 注释）。

---

## 5. 新服务器快速自检

```bash
# OpenSora
conda activate qvdit
python -c "import torch, diffusers, qdiff, opensora; print('qvdit OK', torch.__version__, torch.cuda.get_device_name(0))"

# Wan
conda activate wan-qvdit
python -c "from diffusers import WanPipeline; import qdiff; print('wan-qvdit OK')"
```
