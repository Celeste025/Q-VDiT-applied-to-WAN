#!/usr/bin/env python3
"""Select 25 VBench prompts (8 paper dims), excluding OpenSora calib prompts."""

import argparse
import json
import random
from pathlib import Path

DIMS8 = [
    "imaging_quality",
    "aesthetic_quality",
    "motion_smoothness",
    "dynamic_degree",
    "background_consistency",
    "subject_consistency",
    "scene",
    "overall_consistency",
]


def load_calib_prompts(calib_path: Path) -> set[str]:
    return {line.strip() for line in calib_path.read_text().splitlines() if line.strip()}


def select_subset(vbench_json: Path, calib_path: Path, n: int, seed: int) -> list[dict]:
    calib = load_calib_prompts(calib_path)
    dims8_set = set(DIMS8)
    data = json.loads(vbench_json.read_text())

    candidates = []
    for item in data:
        prompt = item.get("prompt_en") or item.get("prompt")
        item_dims = [d for d in item.get("dimension", []) if d in dims8_set]
        if not prompt or not item_dims or prompt in calib:
            continue
        if len(prompt.encode("utf-8")) > 200:
            continue
        candidates.append(
            {
                "prompt_en": prompt,
                "dimension": item_dims,
                "primary_dimension": item_dims[0],
            }
        )
        if "auxiliary_info" in item:
            candidates[-1]["auxiliary_info"] = item["auxiliary_info"]

    rng = random.Random(seed)
    selected: dict[str, dict] = {}
    per_dim = max(3, n // len(DIMS8))

    for dim in DIMS8:
        pool = [c for c in candidates if dim in c["dimension"] and c["prompt_en"] not in selected]
        rng.shuffle(pool)
        for c in pool[:per_dim]:
            selected[c["prompt_en"]] = c
            if len(selected) >= n:
                break
        if len(selected) >= n:
            break

    remaining = [c for c in candidates if c["prompt_en"] not in selected]
    rng.shuffle(remaining)
    for c in remaining:
        if len(selected) >= n:
            break
        selected[c["prompt_en"]] = c

    items = list(selected.values())[:n]
    if len(items) < n:
        raise RuntimeError(f"Only selected {len(items)} prompts (wanted {n})")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--vbench-json",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "ViDiT-Q-viditq/eval/video/Vbench/vbench/VBench_full_info.json",
    )
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to <root>/logs/eval_vbench_subset25",
    )
    args = parser.parse_args()

    root = args.root
    out_dir = args.out_dir or (root / "logs/eval_vbench_subset25")
    calib_path = root / "t2v/assets/texts/t2v_samples_10.txt"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = select_subset(args.vbench_json, calib_path, args.n, args.seed)
    json_path = out_dir / "vbench_subset_25.json"
    txt_path = out_dir / "prompts.txt"
    meta_path = out_dir / "subset_meta.json"

    json_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
    txt_path.write_text("\n".join(x["prompt_en"] for x in items) + "\n")

    coverage = {d: sum(1 for x in items if d in x["dimension"]) for d in DIMS8}
    meta = {"n": len(items), "seed": args.seed, "coverage": coverage, "excluded_calib": str(calib_path)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Wrote {len(items)} prompts to {json_path}")
    print("Coverage:", coverage)


if __name__ == "__main__":
    main()
