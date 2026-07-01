"""Compare Wan FP16 vs quantized VBench JSON scores."""

from __future__ import annotations

import json
from pathlib import Path


def _extract_score(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return float(value[0])
    return None


def load_scores(results_dir: Path) -> dict[str, float]:
    jsons = sorted(results_dir.glob("*eval_results.json"))
    if not jsons:
        raise FileNotFoundError(f"No eval_results.json under {results_dir}")
    data = json.loads(jsons[0].read_text())
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    scores: dict[str, float] = {}
    for key, value in data.items():
        score = _extract_score(value)
        if score is not None:
            scores[key] = score
    return scores


def compare(fp16_dir: Path, quant_dir: Path) -> str:
    fp = load_scores(fp16_dir)
    q = load_scores(quant_dir)
    lines = ["| dimension | fp16 | quant | delta |", "|---|---:|---:|---:|"]
    for key in sorted(set(fp) & set(q)):
        d = q[key] - fp[key]
        lines.append(f"| {key} | {fp[key]:.4f} | {q[key]:.4f} | {d:+.4f} |")
    return "\n".join(lines)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--fp16-results", required=True)
    p.add_argument("--quant-results", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    table = compare(Path(args.fp16_results), Path(args.quant_results))
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")


if __name__ == "__main__":
    main()
