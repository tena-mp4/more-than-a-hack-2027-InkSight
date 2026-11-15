"""Generate, validate, score, and plot all three approaches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import possession_metrics as pm  # noqa: E402
from solutions.possession_models import (  # noqa: E402
    APPROACHES,
    MODEL_CONFIG,
    build_features,
    resolve_tactical_metadata,
    write_config,
    write_submission,
)


def svg_plot(rows, path: Path) -> None:
    """Write a dependency-free grouped bar chart."""
    width, height = 920, 500
    left, right, top, bottom = 90, 30, 55, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = {"spellF1": "#2563eb", "macroF1": "#f97316"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">Public 5-minute validation</text>'.format(width / 2),
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + plot_h * (1 - value)
        parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#e5e7eb"/>'.format(left, y, width - right, y))
        parts.append('<text x="{}" y="{}" text-anchor="end" font-family="sans-serif" font-size="12">{:.1f}</text>'.format(left - 10, y + 4, value))
    group_w = plot_w / len(rows)
    bar_w = min(70, group_w * 0.28)
    for index, row in enumerate(rows):
        center = left + group_w * (index + 0.5)
        for offset, (metric, key) in enumerate((("spellF1", "spell_f1"), ("macroF1", "macro_f1"))):
            value = float(row[key])
            x = center + (offset - 0.5) * bar_w
            y = top + plot_h * (1 - value)
            parts.append('<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>'.format(x, y, bar_w, top + plot_h - y, colors[metric]))
            parts.append('<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12">{:.3f}</text>'.format(x + bar_w / 2, y - 7, value))
        label = row["approach"].replace("approach_", "A").replace("_", " ")
        parts.append('<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="13">{}</text>'.format(center, height - 70, label))
    legend_x = width - 230
    for idx, metric in enumerate(("spellF1", "macroF1")):
        x = legend_x + idx * 110
        parts.append('<rect x="{}" y="{}" width="14" height="14" fill="{}"/>'.format(x, height - 35, colors[metric]))
        parts.append('<text x="{}" y="{}" font-family="sans-serif" font-size="12">{}</text>'.format(x + 20, height - 23, metric))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results" / "top3")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_tactical_metadata(args.data_dir)
    write_config(args.output_dir / "frozen_config.json", metadata_path)
    print("tactical metadata: {}".format(metadata_path))
    features = build_features(args.data_dir, args.output_dir / "features_cache.npz")

    gt_path = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    gt_table = pm.read_label_csv(str(gt_path))
    gt_codes = pm.label_codes(gt_table)
    rows = []
    for name, model in APPROACHES.items():
        labels = model(features, MODEL_CONFIG)
        submission = args.output_dir / (name + ".csv")
        write_submission(submission, labels)
        table = pm.read_label_csv(str(submission))
        validation_error = pm.validate_submission(table, len(gt_codes))
        if validation_error:
            raise RuntimeError("{}: {}".format(name, validation_error))
        result = pm.evaluate(gt_codes, pm.label_codes(table), bootstrap_n=0)
        row = {
            "approach": name,
            "spell_f1": result["spell_f1_mean"],
            "macro_f1": result["macro_f1"],
            "alive_dead_f1": result["alive_dead_f1"],
            "team_accuracy_alive": result["team_acc_gt_alive"],
            "predicted_possessions": result["n_pred_spells"],
            "mean_possession_seconds": result["mean_spell_duration_s"],
        }
        rows.append(row)
        print("{} spellF1={:.4f} macroF1={:.4f} possessions={}".format(
            name, row["spell_f1"], row["macro_f1"], row["predicted_possessions"]
        ))

    rows.sort(key=lambda row: (-row["spell_f1"], -row["macro_f1"]))
    comparison = args.output_dir / "comparison_metrics.json"
    comparison.write_text(json.dumps(rows, indent=2) + "\n")
    svg_plot(rows, args.output_dir / "comparison.svg")
    print("best on public check: {}".format(rows[0]["approach"]))


if __name__ == "__main__":
    main()
