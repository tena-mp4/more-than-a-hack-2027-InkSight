"""Train, cross-validate and run four DFL-supervised possession approaches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as pm  # noqa: E402
from solutions.dfl_experiments import (  # noqa: E402
    D, BoundaryModel, EmissionModel, apply_smart_tags, cache_dfl, codes_to_labels,
    extract_target_raw, frame_vote, hard_labels, load_raw, make_features, run_ranges,
    save_config, smart_tag_features, spell_vote, viterbi_decode,
)
from solutions.possession_models import approach_state_machine, build_features, write_submission  # noqa: E402


def evaluate_codes(gt, pred, run):
    # DFL arrays contain no X but retain separate half-runs through a one-frame separator here.
    gt_parts, pred_parts = [], []
    for lo, hi in run_ranges(run):
        gt_parts.extend(gt[lo:hi].tolist())
        pred_parts.extend(pred[lo:hi].tolist())
        gt_parts.append(pm.X)
        pred_parts.append(pm.X)
    gt_arr = np.asarray(gt_parts[:-1], dtype=np.int8)
    pred_arr = np.asarray(pred_parts[:-1], dtype=np.int8)
    runs = pm.valid_runs(gt_arr)
    return pm.evaluate(gt_arr, pred_arr, bootstrap_n=0)


def compact(metrics):
    return {
        "spell_f1": metrics["spell_f1_mean"],
        "macro_f1": metrics["macro_f1"],
        "alive_dead_f1": metrics["alive_dead_f1"],
        "n_gt_spells": metrics["n_gt_spells"],
        "n_pred_spells": metrics["n_pred_spells"],
    }


def cross_validate(matches):
    rows = []
    for held_out in range(len(matches)):
        name, x_test, y_test, run_test = matches[held_out]
        train = [(x, y) for i, (_, x, y, _) in enumerate(matches) if i != held_out]
        print("LOMO {} ({}/{})".format(name, held_out + 1, len(matches)), flush=True)
        emission = EmissionModel(seed=2027 + held_out).fit(train)
        boundary = BoundaryModel(seed=3027 + held_out).fit(train)
        probs = emission.predict_proba(x_test)
        bprob = boundary.predict_proba(x_test)
        frame = hard_labels(probs, run_test)
        structured = viterbi_decode(probs, run_ranges(run_test))
        boundary_pred = viterbi_decode(probs, run_ranges(run_test), boundary=bprob)
        voted = spell_vote([frame, structured, boundary_pred], probs, run_test, bprob)
        for approach, prediction in (
            ("01_dfl_frame", frame), ("03_dfl_structured", structured),
            ("04_boundary_fusion", boundary_pred), ("06_spell_vote", voted),
        ):
            row = {"match": name, "approach": approach}
            row.update(compact(evaluate_codes(y_test, prediction, run_test)))
            rows.append(row)
    return rows


def plot_svg(path: Path, rows):
    approaches = [row["approach"] for row in rows]
    spell = [row["spell_f1"] for row in rows]
    macro = [row["macro_f1"] for row in rows]
    width, height = 920, 420
    left, top, chart_h = 80, 45, 285
    group = (width - left - 40) / len(rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="25" font-family="sans-serif" font-size="16">Frozen public diagnostic: spellF1 vs macroF1</text>']
    for tick in range(6):
        value = tick / 5
        y = top + chart_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-30}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="45" y="{y+4:.1f}" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for i, (name, s, m) in enumerate(zip(approaches, spell, macro)):
        x = left + i * group + group * .18
        for offset, value, colour in ((0, s, "#3567b7"), (group * .28, m, "#e28b2d")):
            h = chart_h * value
            parts.append(f'<rect x="{x+offset:.1f}" y="{top+chart_h-h:.1f}" width="{group*.23:.1f}" height="{h:.1f}" fill="{colour}"/>')
        label = name.replace("_", " ")
        parts.append(f'<text x="{left+(i+.5)*group:.1f}" y="{top+chart_h+18}" text-anchor="middle" font-family="sans-serif" font-size="9">{label}</text>')
    parts.extend([
        '<rect x="680" y="16" width="12" height="12" fill="#3567b7"/><text x="697" y="27" font-family="sans-serif" font-size="11">spellF1</text>',
        '<rect x="755" y="16" width="12" height="12" fill="#e28b2d"/><text x="772" y="27" font-family="sans-serif" font-size="11">macroF1</text>',
        '</svg>'])
    path.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--extra-dir", type=Path, default=ROOT / "data" / "extra" / "28196177")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results" / "dfl_four")
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = cache_dfl(args.extra_dir, args.output_dir / "dfl_cache")
    matches = []
    for path in cache_paths:
        raw = load_raw(path)
        matches.append((path.stem, make_features(raw), raw["labels"], raw["run"]))

    cv_rows = [] if args.skip_cv else cross_validate(matches)
    if not args.skip_cv:
        (args.output_dir / "dfl_cv.json").write_text(json.dumps(cv_rows, indent=2) + "\n")

    print("fitting final seven-match models", flush=True)
    train = [(x, y) for _, x, y, _ in matches]
    emission = EmissionModel().fit(train)
    boundary_model = BoundaryModel().fit(train)
    target_raw = extract_target_raw(args.data_dir)
    target_x = make_features(target_raw)
    probs = emission.predict_proba(target_x)
    bprob = boundary_model.predict_proba(target_x)
    tags = smart_tag_features(args.data_dir, len(target_x))
    tagged_probs = apply_smart_tags(probs, tags)
    runs = run_ranges(target_raw["run"])

    raw_frame = hard_labels(probs, target_raw["run"])
    pred1 = viterbi_decode(probs, runs)
    pred2 = viterbi_decode(tagged_probs, runs)
    pred3 = viterbi_decode(probs, runs, boundary=bprob)
    combined_boundary = np.maximum(bprob, 0.55 * tags["boundary"] + 0.35 * tags["transition"])
    pred4 = viterbi_decode(tagged_probs, runs, boundary=combined_boundary)

    # Include the independent football-rule model in voting, but not among the four learned
    # approaches: diversity is useful to an ensemble even when the individual model is weaker.
    legacy_features = build_features(args.data_dir, args.output_dir / "legacy_features.npz")
    legacy_labels = approach_state_machine(legacy_features)
    legacy = np.full(len(legacy_labels), D, dtype=np.int8)
    for code, text in enumerate(("H", "A", "D")):
        legacy[legacy_labels == text] = code

    frame_voted = frame_vote([pred1, pred2, pred3, pred4, legacy], tagged_probs)
    spell_voted = spell_vote([pred1, pred2, pred3, pred4, legacy], tagged_probs,
                             target_raw["run"], combined_boundary)
    outputs = {
        "00_raw_frame_ablation": raw_frame,
        "01_dfl_structured": pred1,
        "02_semantic_tags": pred2,
        "03_learned_boundary": pred3,
        "04_phase_boundary": pred4,
        "05_frame_vote": frame_voted,
        "06_spell_vote": spell_voted,
    }
    for name, codes in outputs.items():
        write_submission(args.output_dir / (name + ".csv"), codes_to_labels(codes, target_raw["run"]))

    # Only now read the public labels. No result above can depend on them.
    gt = pm.label_codes(pm.read_label_csv(ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"))
    public_rows = []
    public_detailed = {}
    for name in outputs:
        pred = pm.label_codes(pm.read_label_csv(args.output_dir / (name + ".csv")))
        detailed = pm.evaluate(gt, pred, bootstrap_n=0)
        public_detailed[name] = detailed
        row = {"approach": name}
        row.update(compact(detailed))
        public_rows.append(row)
    (args.output_dir / "public_results.json").write_text(json.dumps(public_rows, indent=2) + "\n")
    (args.output_dir / "public_detailed.json").write_text(
        json.dumps(public_detailed, indent=2) + "\n")
    save_config(args.output_dir / "frozen_config.json", cache_paths)
    plot_svg(args.output_dir / "comparison.svg", public_rows)
    print(json.dumps(public_rows, indent=2))


if __name__ == "__main__":
    main()
