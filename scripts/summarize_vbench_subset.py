#!/usr/bin/env python3
"""Summarize VBench subset-25 scores for FP16 and quant models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "logs/eval_vbench_subset25"

MODELS = [
    ("fp16", EVAL_DIR / "vbench_results"),
    ("w8a8", EVAL_DIR / "vbench_results_w8a8"),
    ("w4a6", EVAL_DIR / "vbench_results_w4a6"),
    ("w4a4", EVAL_DIR / "vbench_results_w4a4"),
]

DIM_ORDER = [
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "scene",
]


def extract_score(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return float(value[0])
    return None


def load_scores(result_dir: Path) -> dict[str, float] | None:
    jsons = sorted(result_dir.glob("*eval_results.json"))
    if not jsons:
        return None
    data = json.loads(jsons[0].read_text())
    scores: dict[str, float] = {}
    for key, value in data.items():
        s = extract_score(value)
        if s is not None:
            scores[key] = s
    return scores


def main() -> int:
    out_md = EVAL_DIR / "vbench_subset_summary.md"
    rows: list[tuple[str, dict[str, float]]] = []
    missing: list[str] = []

    for name, path in MODELS:
        scores = load_scores(path)
        if scores is None:
            missing.append(name)
            continue
        rows.append((name, scores))

    if not rows:
        print("No VBench results found.", file=sys.stderr)
        return 1

    dims = [d for d in DIM_ORDER if any(d in sc for _, sc in rows)]
    for _, sc in rows:
        for k in sc:
            if k not in dims:
                dims.append(k)

    header = "| dimension | " + " | ".join(n for n, _ in rows) + " |"
    sep = "|---|" + "|".join(["---:"] * len(rows)) + "|"
    lines = [
        "# VBench Subset-25 Summary (OpenSora 512×512, 100 steps)",
        "",
        header,
        sep,
    ]
    fp16_scores = dict(rows).get("fp16", {})
    for dim in dims:
        cells = []
        for name, sc in rows:
            if dim not in sc:
                cells.append("—")
            else:
                val = sc[dim]
                if name != "fp16" and dim in fp16_scores:
                    delta = val - fp16_scores[dim]
                    cells.append(f"{val:.4f} ({delta:+.4f})")
                else:
                    cells.append(f"{val:.4f}")
        lines.append(f"| {dim} | " + " | ".join(cells) + " |")

    if missing:
        lines.extend(["", f"Pending / missing: {', '.join(missing)}"])

    text = "\n".join(lines) + "\n"
    out_md.write_text(text)
    print(text)
    print(f"Saved -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
