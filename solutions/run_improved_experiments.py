"""Benchmark five research-screened possession improvements.

Selection uses only leave-one-match-out DFL validation after applying contiguous missing-ball gaps
sampled from the unlabelled challenge input.  Public labels are loaded only after final predictions
are written.
"""

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
    BoundaryModel, EmissionModel, apply_smart_tags, boundary_veto_decode, cache_dfl,
    codes_to_labels, extract_target_raw, hierarchical_alive_calibrate, hierarchical_decode, load_raw,
    make_context_features, make_features, mask_raw_contiguous, missing_gap_lengths,
    run_ranges, smart_tag_features, sparse_boundary_decode, viterbi_decode,
)
from solutions.possession_models import write_submission  # noqa: E402
from solutions.run_dfl_experiments import compact, evaluate_codes, plot_svg  # noqa: E402


HIERARCHICAL_GRID = (
    (-1.50, 6.0), (-1.00, 6.0), (-0.50, 6.0),
    (-1.00, 8.0), (-0.50, 8.0), (0.00, 8.0),
)
SPARSE_GRID = (
    (0.40, 5.0, 14.0), (0.60, 5.0, 14.0), (0.80, 5.0, 14.0),
    (0.60, 6.0, 16.0), (0.80, 6.0, 16.0),
)
VETO_GRID = ((0.30, 6), (0.50, 8), (0.70, 8), (0.80, 12))


def prepare_matches(cache_paths, gap_lengths):
    matches = []
    for index, path in enumerate(cache_paths):
        raw = load_raw(path)
        masked_raw = mask_raw_contiguous(raw, gap_lengths, seed=5000 + index)
        basic = make_features(raw)
        masked = make_features(masked_raw)
        context = make_context_features(basic, raw["run"])
        masked_context = make_context_features(masked, raw["run"])
        matches.append({
            "name": path.stem, "labels": raw["labels"], "run": raw["run"],
            "basic": basic, "masked": masked,
            "context": context, "masked_context": masked_context,
        })
    return matches


def _record(rows, match, method, config, metrics):
    row = {"match": match, "method": method, "config": config}
    row.update(compact(metrics))
    rows.append(row)


def cross_validate(matches):
    rows = []
    for held, test in enumerate(matches):
        print("improved LOMO {} ({}/{})".format(test["name"], held + 1, len(matches)), flush=True)
        basic_train, context_train = [], []
        for index, match in enumerate(matches):
            if index == held:
                continue
            basic_train.extend(((match["basic"], match["labels"]),
                                (match["masked"], match["labels"])))
            context_train.extend(((match["context"], match["labels"]),
                                  (match["masked_context"], match["labels"])))
        basic_model = EmissionModel(seed=6100 + held).fit(basic_train, independent_mask=False)
        context_model = EmissionModel(seed=7100 + held).fit(context_train, independent_mask=False)
        boundary_model = BoundaryModel(seed=8100 + held).fit(context_train)
        p_basic = basic_model.predict_proba(test["masked"])
        p_context = context_model.predict_proba(test["masked_context"])
        bprob = boundary_model.predict_proba(test["masked_context"])
        runs = run_ranges(test["run"])

        prediction = viterbi_decode(p_basic, runs)
        _record(rows, test["name"], "01_gap_augmented", "cost=8",
                evaluate_codes(test["labels"], prediction, test["run"]))
        prediction = viterbi_decode(p_context, runs)
        _record(rows, test["name"], "02_temporal_context", "cost=8",
                evaluate_codes(test["labels"], prediction, test["run"]))

        for bias, dead_cost in HIERARCHICAL_GRID:
            prediction = hierarchical_decode(p_context, runs, dead_bias=bias,
                                               dead_switch_cost=dead_cost)
            config = "bias={:.2f},dead_cost={:.1f}".format(bias, dead_cost)
            _record(rows, test["name"], "03_hierarchical", config,
                    evaluate_codes(test["labels"], prediction, test["run"]))
        calibrated = hierarchical_alive_calibrate(p_context, runs)
        for threshold, candidate_cost, other_cost in SPARSE_GRID:
            prediction = sparse_boundary_decode(
                calibrated, runs, bprob, threshold, candidate_cost, other_cost)
            config = "threshold={:.2f},candidate={:.1f},other={:.1f}".format(
                threshold, candidate_cost, other_cost)
            _record(rows, test["name"], "04_sparse_boundary", config,
                    evaluate_codes(test["labels"], prediction, test["run"]))
        for threshold, radius in VETO_GRID:
            prediction = boundary_veto_decode(p_context, runs, bprob, threshold, radius)
            config = "threshold={:.2f},radius={}".format(threshold, radius)
            _record(rows, test["name"], "05_boundary_veto", config,
                    evaluate_codes(test["labels"], prediction, test["run"]))
    return rows


def select_configs(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["config"])].append(row["spell_f1"])
    selected = {}
    summary = []
    for (method, config), scores in sorted(grouped.items()):
        summary.append({"method": method, "config": config,
                        "mean_spell_f1": float(np.mean(scores)),
                        "std_spell_f1": float(np.std(scores))})
    for method in sorted({row["method"] for row in rows}):
        candidates = [row for row in summary if row["method"] == method]
        best = max(candidates, key=lambda row: (row["mean_spell_f1"], row["config"]))
        selected[method] = best["config"]
    return selected, summary


def parse_config(text):
    out = {}
    for item in text.split(","):
        key, value = item.split("=")
        out[key] = float(value)
    return out


def final_models(matches, target_raw, selected, data_dir):
    basic_train, context_train = [], []
    for match in matches:
        basic_train.extend(((match["basic"], match["labels"]),
                            (match["masked"], match["labels"])))
        context_train.extend(((match["context"], match["labels"]),
                              (match["masked_context"], match["labels"])))
    basic_model = EmissionModel(seed=6100).fit(basic_train, independent_mask=False)
    context_model = EmissionModel(seed=7100).fit(context_train, independent_mask=False)
    boundary_model = BoundaryModel(seed=8100).fit(context_train)
    basic = make_features(target_raw)
    context = make_context_features(basic, target_raw["run"])
    p_basic = basic_model.predict_proba(basic)
    p_context = context_model.predict_proba(context)
    tags = smart_tag_features(data_dir, len(basic))
    tagged = apply_smart_tags(p_context, tags)
    bprob = boundary_model.predict_proba(context)
    runs = run_ranges(target_raw["run"])

    outputs = {}
    outputs["01_gap_augmented"] = viterbi_decode(p_basic, runs)
    outputs["02_temporal_semantic"] = viterbi_decode(tagged, runs)
    cfg = parse_config(selected["03_hierarchical"])
    outputs["03_hierarchical"] = hierarchical_decode(
        tagged, runs, dead_bias=cfg["bias"], dead_switch_cost=cfg["dead_cost"])
    cfg = parse_config(selected["04_sparse_boundary"])
    calibrated = hierarchical_alive_calibrate(tagged, runs)
    outputs["04_sparse_boundary"] = sparse_boundary_decode(
        calibrated, runs, bprob, cfg["threshold"], cfg["candidate"], cfg["other"])
    cfg = parse_config(selected["05_boundary_veto"])
    outputs["05_boundary_veto"] = boundary_veto_decode(
        tagged, runs, bprob, cfg["threshold"], int(cfg["radius"]))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--extra-dir", type=Path,
                        default=ROOT / "data" / "extra" / "28196177")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "evaluation" / "results" / "improved_five")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = cache_dfl(args.extra_dir, ROOT / "evaluation" / "results" / "dfl_four" / "dfl_cache")
    target_raw = extract_target_raw(args.data_dir)
    gaps = missing_gap_lengths(target_raw["visible"], target_raw["run"])
    matches = prepare_matches(cache_paths, gaps)

    rows = cross_validate(matches)
    selected, summary = select_configs(rows)
    (args.output_dir / "cv_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "selected_config.json").write_text(json.dumps({
        "selected": selected,
        "target_gap_statistics": {
            "n_gaps": int(len(gaps)), "median_frames": float(np.median(gaps)),
            "p90_frames": float(np.percentile(gaps, 90)),
            "max_frames": int(np.max(gaps)),
        },
        "public_ground_truth_used_for_selection": False,
    }, indent=2) + "\n")
    print("selected {}".format(json.dumps(selected, sort_keys=True)), flush=True)

    outputs = final_models(matches, target_raw, selected, args.data_dir)
    for name, codes in outputs.items():
        write_submission(args.output_dir / (name + ".csv"),
                         codes_to_labels(codes, target_raw["run"]))

    # Frozen diagnostic: this is deliberately the first public-ground-truth read in the script.
    gt = pm.label_codes(pm.read_label_csv(
        ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"))
    public_rows, detailed = [], {}
    for name in outputs:
        pred = pm.label_codes(pm.read_label_csv(args.output_dir / (name + ".csv")))
        result = pm.evaluate(gt, pred, bootstrap_n=0)
        detailed[name] = result
        row = {"approach": name}
        row.update(compact(result))
        public_rows.append(row)
    (args.output_dir / "public_results.json").write_text(json.dumps(public_rows, indent=2) + "\n")
    (args.output_dir / "public_detailed.json").write_text(json.dumps(detailed, indent=2) + "\n")
    plot_svg(args.output_dir / "comparison.svg", public_rows)
    print(json.dumps(public_rows, indent=2))


if __name__ == "__main__":
    main()
