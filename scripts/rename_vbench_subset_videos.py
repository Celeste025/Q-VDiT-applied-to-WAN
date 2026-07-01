#!/usr/bin/env python3
"""Copy sample_*.mp4 from quant raw output to VBench prompt-named videos."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def resolve_raw_dir(eval_dir: Path, tag: str) -> Path:
    """Find sample_*.mp4 directory (handles legacy _opensora_opensora bug)."""
    candidates = (
        eval_dir / f"{tag}_videos_opensora",
        eval_dir / f"{tag}_videos_opensora_opensora",
    )
    for path in candidates:
        if any(path.glob("sample_*.mp4")):
            return path
    return candidates[0]


def rename_subset_videos(eval_dir: Path, tag: str) -> Path:
    eval_dir = eval_dir.resolve()
    prompts_path = eval_dir / "prompts.txt"
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Missing prompts file: {prompts_path}")

    prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
    raw_dir = resolve_raw_dir(eval_dir, tag)
    out_dir = eval_dir / f"{tag}_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not any(raw_dir.glob("sample_*.mp4")):
        raise FileNotFoundError(f"No sample_*.mp4 under {raw_dir}")

    for i, prompt in enumerate(prompts):
        src = raw_dir / f"sample_{i}.mp4"
        dst = out_dir / f"{prompt}.mp4"
        if not src.is_file():
            raise FileNotFoundError(f"missing generated video: {src}")
        shutil.copy2(src, dst)
        print(f"[ok] {dst.name}")

    count = len(list(out_dir.glob("*.mp4")))
    print(f"Renamed {len(prompts)} videos from {raw_dir} -> {out_dir} ({count} mp4 total)")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="e.g. w4a6, w8a8")
    parser.add_argument("--eval-dir", required=True)
    args = parser.parse_args()
    try:
        rename_subset_videos(Path(args.eval_dir), args.tag)
    except (FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
