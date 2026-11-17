"""Second, leakage-safe iteration of the sparse possession-boundary decoder."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as pm  # noqa: E402
from solutions.dfl_experiments import (  # noqa: E402
    BoundaryModel, EmissionModel, apply_smart_tags, cache_dfl, codes_to_labels,
    extract_target_raw, hierarchical_alive_calibrate, load_raw, make_context_features, make_features,
    mask_raw_contiguous, missing_gap_lengths, run_ranges, smart_tag_features,
    sparse_boundary_decode, sparse_peak_boundary_decode,
)
from solutions.possession_models import write_submission  # noqa: E402
from solutions.run_dfl_experiments import compact, plot_svg  # noqa: E402
from solutions.run_improved_experiments import evaluate_codes  # noqa: E402


# Compact predeclared grid: threshold, smoothing radius, switch-window radius,
# peak separation, minimum team run, minimum dead run. Transition costs remain
# frozen at the first iteration's DFL-selected values (5 and 14).
CONFIGS = (
    (0.55, 1, 3, 12, 0, 0),
    (0.65, 1, 4, 12, 0, 0),
    (0.75, 1, 4, 12, 0, 0),
    (0.55, 2, 6, 20, 10, 8),
    (0.65, 2, 6, 20, 10, 8),
    (0.75, 2, 6, 20, 10, 8),
    (0.55, 3, 8, 30, 20, 12),
    (0.65, 3, 8, 30, 20, 12),
    (0.75, 3, 8, 30, 20, 12),
)


def config_dict(values):
    keys = ("threshold", "smooth_radius", "gate_radius", "min_separation",
            "team_min", "dead_min")
    return dict(zip(keys, values))


def prepare(cache_paths, gaps):
    matches = []
    for index, path in enumerate(cache_paths):
        raw = load_raw(path)
        masked = mask_raw_contiguous(raw, gaps, seed=5000 + index)
        matches.append({
            "name": path.stem,
            "labels": raw["labels"],
            "run": raw["run"],
            "clean": make_context_features(make_features(raw), raw["run"]),
            "masked": make_context_features(make_features(masked), raw["run"]),
        })
    return matches


def decode_peak(probs, runs, boundary, config):
    return sparse_peak_boundary_decode(
        probs, runs, boundary, threshold=config["threshold"],
        candidate_cost=5.0, other_cost=14.0,
        smooth_radius=int(config["smooth_radius"]),
        gate_radius=int(config["gate_radius"]),
        min_separation=int(config["min_separation"]),
        team_min=int(config["team_min"]), dead_min=int(config["dead_min"]),
    )


def cross_validate(matches):
    rows = []
    for held, test in enumerate(matches):
        print("sparse iteration LOMO {} ({}/{})".format(
            test["name"], held + 1, len(matches)), flush=True)
        train = []
        for index, match in enumerate(matches):
            if index != held:
                train.extend(((match["clean"], match["labels"]),
                              (match["masked"], match["labels"])))
        # Inherit v1's estimator seeds so the baseline is a true paired control.
        emission = EmissionModel(seed=7100 + held).fit(train, independent_mask=False)
        boundary_model = BoundaryModel(seed=8100 + held).fit(train)
        probs = emission.predict_proba(test["masked"])
        boundary = boundary_model.predict_proba(test["masked"])
        runs = run_ranges(test["run"])
        probs = hierarchical_alive_calibrate(probs, runs)
        baseline = sparse_boundary_decode(probs, runs, boundary, 0.8, 5.0, 14.0)
        metrics = evaluate_codes(test["labels"], baseline, test["run"])
        row = {"match": test["name"], "method": "baseline_sparse_v1",
               "config": "threshold=0.8,candidate=5,other=14"}
        row.update(compact(metrics))
        rows.append(row)
        for values in CONFIGS:
            config = config_dict(values)
            prediction = decode_peak(probs, runs, boundary, config)
            metrics = evaluate_codes(test["labels"], prediction, test["run"])
            row = {"match": test["name"], "method": "peak_duration_v2",
                   "config": config}
            row.update(compact(metrics))
            rows.append(row)
    return rows


def select_config(rows):
    grouped = defaultdict(list)
    configs = {}
    for row in rows:
        if row["method"] != "peak_duration_v2":
            continue
        key = json.dumps(row["config"], sort_keys=True)
        grouped[key].append(row["spell_f1"])
        configs[key] = row["config"]
    summary = [{
        "config": configs[key],
        "mean_spell_f1": float(np.mean(scores)),
        "std_spell_f1": float(np.std(scores)),
    } for key, scores in grouped.items()]
    selected = max(summary, key=lambda row: (row["mean_spell_f1"],
                                              json.dumps(row["config"], sort_keys=True)))
    return selected, sorted(summary, key=lambda row: row["mean_spell_f1"], reverse=True)


def fit_final(matches, target_raw, data_dir, selected):
    train = []
    for match in matches:
        train.extend(((match["clean"], match["labels"]),
                      (match["masked"], match["labels"])))
    emission = EmissionModel(seed=7100).fit(train, independent_mask=False)
    boundary_model = BoundaryModel(seed=8100).fit(train)
    context = make_context_features(make_features(target_raw), target_raw["run"])
    probs = emission.predict_proba(context)
    probs = apply_smart_tags(probs, smart_tag_features(data_dir, len(probs)))
    boundary = boundary_model.predict_proba(context)
    runs = run_ranges(target_raw["run"])
    probs = hierarchical_alive_calibrate(probs, runs)
    return {
        "baseline_sparse_v1": sparse_boundary_decode(
            probs, runs, boundary, 0.8, 5.0, 14.0),
        "peak_duration_v2": decode_peak(probs, runs, boundary, selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--extra-dir", type=Path,
                        default=ROOT / "data" / "extra" / "28196177")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "evaluation" / "results" / "sparse_iteration_v2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = cache_dfl(
        args.extra_dir, ROOT / "evaluation" / "results" / "dfl_four" / "dfl_cache")
    target_raw = extract_target_raw(args.data_dir)
    gaps = missing_gap_lengths(target_raw["visible"], target_raw["run"])
    matches = prepare(cache_paths, gaps)
    rows = cross_validate(matches)
    selected, summary = select_config(rows)
    baseline_cv = float(np.mean([
        row["spell_f1"] for row in rows if row["method"] == "baseline_sparse_v1"
    ]))
    (args.output_dir / "cv_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    selection = {
        "selected": selected,
        "baseline_mean_spell_f1": baseline_cv,
        "accepted_as_improvement": selected["mean_spell_f1"] > baseline_cv,
        "deployed_model": ("peak_duration_v2" if selected["mean_spell_f1"] > baseline_cv
                           else "baseline_sparse_v1"),
        "public_ground_truth_used_for_selection": False,
        "transition_costs_inherited_from_v1": {"candidate": 5.0, "other": 14.0},
    }
    (args.output_dir / "selected_config.json").write_text(
        json.dumps(selection, indent=2) + "\n")
    print("selected {}".format(json.dumps(selected, sort_keys=True)), flush=True)

    outputs = fit_final(matches, target_raw, args.data_dir, selected["config"])
    for name, codes in outputs.items():
        write_submission(args.output_dir / (name + ".csv"),
                         codes_to_labels(codes, target_raw["run"]))

    # The labelled five minutes are opened only after selection and both predictions are frozen.
    gt = pm.label_codes(pm.read_label_csv(
        ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"))
    public_rows, details = [], {}
    for name in outputs:
        pred = pm.label_codes(pm.read_label_csv(args.output_dir / (name + ".csv")))
        result = pm.evaluate(gt, pred, bootstrap_n=0)
        details[name] = result
        row = {"approach": name}
        row.update(compact(result))
        public_rows.append(row)
    (args.output_dir / "public_results.json").write_text(
        json.dumps(public_rows, indent=2) + "\n")
    (args.output_dir / "public_detailed.json").write_text(
        json.dumps(details, indent=2) + "\n")
    plot_svg(args.output_dir / "comparison.svg", public_rows)
    print(json.dumps(public_rows, indent=2))


if __name__ == "__main__":
    main()
