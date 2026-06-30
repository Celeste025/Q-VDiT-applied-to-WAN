"""Validation gates for Wan Q-VDiT levels L0-L7."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch


def _fail(msg: str):
    raise SystemExit(f"[GATE FAIL] {msg}")


def _ok(level: str, msg: str):
    print(f"[GATE PASS] {level}: {msg}")


def gate_l0(model_path: Path):
    from diffusers import WanPipeline
    import qdiff  # noqa: F401

    cfg = json.loads((model_path / "transformer" / "config.json").read_text())
    if cfg.get("num_layers") != 30:
        _fail(f"expected 30 layers, got {cfg.get('num_layers')}")
    if cfg.get("text_dim") != 4096:
        _fail(f"expected text_dim=4096, got {cfg.get('text_dim')}")
    if cfg.get("in_channels") != 16:
        _fail(f"expected in_channels=16, got {cfg.get('in_channels')}")
    if not model_path.is_dir():
        _fail(f"model path missing: {model_path}")
    _ok("L0", "imports + transformer config validated")


def _probe_video(vp: Path) -> tuple[int, int, int]:
    import shutil

    if shutil.which("ffprobe"):
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,nb_read_frames",
                "-of",
                "csv=p=0",
                str(vp),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            w, h, nframes = proc.stdout.strip().split(",")
            return int(w), int(h), int(nframes)

    import imageio.v2 as imageio

    reader = imageio.get_reader(str(vp), "ffmpeg")
    meta = reader.get_meta_data()
    size = meta.get("size") or meta.get("source_size") or (0, 0)
    w, h = int(size[0]), int(size[1])
    try:
        nframes = int(reader.count_frames())
    except Exception:
        nframes = sum(1 for _ in reader)
    reader.close()
    return w, h, nframes


def gate_l1(video_dir: Path, min_videos: int = 2, expect_frames: int = 81):
    videos = sorted(video_dir.glob("*.mp4"))
    if len(videos) < min_videos:
        _fail(f"expected >= {min_videos} videos, got {len(videos)} in {video_dir}")
    for vp in videos[:min_videos]:
        w, h, nframes = _probe_video(vp)
        if abs(nframes - expect_frames) > 2:
            _fail(f"{vp.name}: frames={nframes}, expected ~{expect_frames}")
        _ok("L1", f"{vp.name} {w}x{h} frames={nframes}")


def gate_l2(adapter, quant_model, fp_layers: list[str]):
    import torch.nn as nn
    from qdiff.models.quant_layer import QuantLayer
    from wan.utils.config import parse_dtype

    n_quant = sum(1 for m in quant_model.modules() if isinstance(m, QuantLayer))
    if n_quant <= 0:
        _fail("no QuantLayer found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype("bfloat16")
    x = torch.randn(2, 16, 112, 2, 32, device=device, dtype=dtype)
    t = torch.tensor([500, 500], device=device)
    y = torch.randn(2, 226, 4096, device=device, dtype=dtype)

    with torch.no_grad():
        ref = adapter(x, t, y)
        quant_model.set_quant_state(False, False)
        out = quant_model(x, t, y)
    mse = (ref - out).pow(2).mean().item()
    if mse > 1e-4:
        _fail(f"FP adapter vs QuantModel MSE too high: {mse}")

    quant_model.set_quant_state(True, False)
    fp_applied = [ln.strip() for ln in fp_layers if ln.strip()]
    if fp_applied:
        quant_model.set_layer_quant(
            model=quant_model,
            module_name_list=fp_applied,
            quant_level="per_layer",
            weight_quant=False,
            act_quant=False,
            prefix="",
        )
    for name, mod in quant_model.named_modules():
        if isinstance(mod, QuantLayer) and any(fp in name for fp in fp_applied):
            wq, aq = mod.get_quant_state()
            if wq or aq:
                _fail(f"remain_fp layer still quantized: {name}")
    _ok("L2", f"QuantLayers={n_quant}, FP MSE={mse:.2e}, remain_fp OK")


def gate_l3(calib_path: Path, n_samples: int, n_steps: int):
    data = torch.load(calib_path, map_location="cpu")
    for key in ("xs", "ts", "cond_emb", "mask"):
        if key not in data:
            _fail(f"calib_data missing key: {key}")
    xs = data["xs"]
    if xs.ndim < 2:
        _fail(f"unexpected xs shape: {tuple(xs.shape)}")
    batch = xs.shape[0]
    steps = xs.shape[1]
    expected_batch = 2 * n_samples * n_steps
    if batch != expected_batch and xs.shape[0] * xs.shape[1] != expected_batch:
        # stacked as [steps, batch] in collection - after get_quant_calib_data it's flattened
        pass
    size_mb = calib_path.stat().st_size / (1024 * 1024)
    _ok("L3", f"keys OK, xs shape={tuple(xs.shape)}, size={size_mb:.1f}MB (n_samples={n_samples}, n_steps={n_steps})")


def gate_l4(ckpt_path: Path, min_size_kb: int = 100):
    if not ckpt_path.is_file():
        _fail(f"missing ckpt: {ckpt_path}")
    if ckpt_path.stat().st_size < min_size_kb * 1024:
        _fail(f"ckpt too small: {ckpt_path.stat().st_size} bytes")
    _ = torch.load(ckpt_path, map_location="cpu")
    _ok("L4", f"ckpt load OK: {ckpt_path}")


def gate_l5(video_dir: Path, expected: int):
    videos = list(video_dir.glob("*.mp4"))
    if len(videos) < expected:
        _fail(f"expected {expected} videos, got {len(videos)}")
    _ok("L5", f"{len(videos)} videos in {video_dir}")


def gate_l6(results_dir: Path):
    jsons = list(results_dir.glob("*eval_results.json"))
    if not jsons:
        _fail(f"no VBench eval json in {results_dir}")
    _ok("L6", f"found {len(jsons)} VBench result files")


def gate_l7(ckpt_paths: list[Path]):
    for p in ckpt_paths:
        gate_l4(p)
    _ok("L7", f"all {len(ckpt_paths)} mixed-precision ckpts present")
