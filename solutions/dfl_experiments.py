"""DFL-supervised possession experiments.

The public five-minute labels are never imported while fitting a model or choosing a
configuration.  They are read only by ``run_dfl_experiments.py`` after every model has produced
its frozen full-match prediction.

This module intentionally stays compact: one streaming DFL XML reader, one feature definition,
two small histogram-gradient-boosting classifiers, and deterministic sequence decoders.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from games import GAME, GAMES, half_video_blocks
from solutions.possession_models import (
    LABELS,
    MODEL_CONFIG,
    add_tag_features,
    approach_state_machine,
    build_features as build_legacy_features,
    resolve_tactical_metadata,
    team_of_tracks,
)
from starter.load_smart_tagging import load as load_smart_tagging


H, A, D = 0, 1, 2
FPS = 25
DFL_GLOB = "DFL_04_03_positions_raw_observed_*.xml"
DFL_CACHE_SCHEMA = "dfl-features-v1"


@dataclass(frozen=True)
class ExperimentConfig:
    sample_stride: int = 5
    random_seed: int = 2027
    missing_fraction: float = 0.41
    tag_team_weight: float = 0.45
    tag_dead_weight: float = 0.20
    # Selected from the fixed [2.2, 3, 4, 5, 6, 7, 8] grid by seven-match leave-one-out DFL
    # spellF1.  The public five-minute labels were not loaded during this selection.
    viterbi_switch_cost: float = 8.0
    boundary_switch_discount: float = 2.0
    min_team_run: int = 3
    min_dead_run: int = 5
    vote_tolerance: int = 12


CONFIG = ExperimentConfig()


def _match_id(path: Path) -> str:
    return path.stem.rsplit("-", 1)[-1]


def _match_metadata(position_path: Path) -> Tuple[Path, str, str, float, float]:
    match_id = _match_id(position_path)
    candidates = list(position_path.parent.glob("DFL_02_01_matchinformation_*{}*.xml".format(match_id)))
    if len(candidates) != 1:
        raise ValueError("expected one metadata file for {}, got {}".format(match_id, candidates))
    root = ET.parse(candidates[0]).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    if general is None or environment is None:
        raise ValueError("missing General/Environment in {}".format(candidates[0]))
    return (candidates[0], general.get("HomeTeamId"), general.get("GuestTeamId"),
            float(environment.get("PitchX")), float(environment.get("PitchY")))


def _frameset_arrays(elem: ET.Element, ball: bool = False) -> Dict[str, np.ndarray]:
    frames = list(elem)
    out = {
        "n": np.fromiter((int(x.get("N")) for x in frames), dtype=np.int64, count=len(frames)),
        "x": np.fromiter((float(x.get("X")) for x in frames), dtype=np.float32, count=len(frames)),
        "y": np.fromiter((float(x.get("Y")) for x in frames), dtype=np.float32, count=len(frames)),
        "speed": np.fromiter((float(x.get("S", "nan")) / 3.6 for x in frames),
                             dtype=np.float32, count=len(frames)),
    }
    if ball:
        out.update({
            "z": np.fromiter((float(x.get("Z", "nan")) for x in frames),
                             dtype=np.float32, count=len(frames)),
            "status": np.fromiter((int(x.get("BallStatus")) for x in frames),
                                  dtype=np.int8, count=len(frames)),
            "possession": np.fromiter((int(x.get("BallPossession")) for x in frames),
                                      dtype=np.int8, count=len(frames)),
        })
    return out


def extract_dfl_match(position_path: Path) -> Dict[str, np.ndarray]:
    """Read one large DFL XML without constructing the full XML tree in memory."""
    _, home_id, away_id, pitch_length, pitch_width = _match_metadata(position_path)
    players: Dict[str, List[Tuple[str, Dict[str, np.ndarray]]]] = {}
    balls: Dict[str, Dict[str, np.ndarray]] = {}

    for _, elem in ET.iterparse(str(position_path), events=("end",)):
        if elem.tag != "FrameSet":
            continue
        section = elem.get("GameSection")
        team_id = elem.get("TeamId")
        if team_id == "BALL":
            balls[section] = _frameset_arrays(elem, ball=True)
        elif team_id in (home_id, away_id):
            players.setdefault(section, []).append((team_id, _frameset_arrays(elem)))
        elem.clear()

    sections = []
    for section in ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra"):
        if section not in balls:
            continue
        ball = balls[section]
        n = len(ball["n"])
        n0 = int(ball["n"][0])
        raw = _empty_raw(n)
        raw["visible"][:] = True
        raw["ball_x"] = ball["x"]
        raw["ball_y"] = ball["y"]
        raw["ball_z"] = ball["z"]
        raw["ball_speed"] = ball["speed"]
        raw["pitch_length"][:] = pitch_length
        raw["pitch_width"][:] = pitch_width

        for team_id, player in players.get(section, []):
            idx = player["n"] - n0
            good = (idx >= 0) & (idx < n)
            idx = idx[good].astype(np.int64)
            dx = player["x"][good] - ball["x"][idx]
            dy = player["y"][good] - ball["y"][idx]
            dist = np.hypot(dx, dy).astype(np.float32)
            side = "home" if team_id == home_id else "away"
            _update_team_geometry(raw, side, idx, dist, player["speed"][good])

        labels = np.where(ball["status"] == 0, D,
                          np.where(ball["possession"] == 1, H, A)).astype(np.int8)
        raw["labels"] = labels
        raw["run"] = np.full(n, len(sections), dtype=np.int16)
        sections.append(raw)

    if not sections:
        raise ValueError("no ball FrameSet in {}".format(position_path))
    return _concatenate_raw(sections)


def _empty_raw(n: int) -> Dict[str, np.ndarray]:
    out = {
        "visible": np.zeros(n, dtype=bool),
        "ball_x": np.full(n, np.nan, dtype=np.float32),
        "ball_y": np.full(n, np.nan, dtype=np.float32),
        "ball_z": np.full(n, np.nan, dtype=np.float32),
        "ball_speed": np.full(n, np.nan, dtype=np.float32),
        "pitch_length": np.full(n, np.nan, dtype=np.float32),
        "pitch_width": np.full(n, np.nan, dtype=np.float32),
    }
    for side in ("home", "away"):
        out[side + "_d1"] = np.full(n, np.inf, dtype=np.float32)
        out[side + "_d2"] = np.full(n, np.inf, dtype=np.float32)
        out[side + "_arrival"] = np.full(n, np.inf, dtype=np.float32)
        for radius in (1, 3, 5):
            out["{}_count{}".format(side, radius)] = np.zeros(n, dtype=np.int8)
    return out


def _update_team_geometry(raw: Dict[str, np.ndarray], side: str, idx: np.ndarray,
                          distance: np.ndarray, speed: np.ndarray) -> None:
    first = raw[side + "_d1"][idx]
    second = raw[side + "_d2"][idx]
    raw[side + "_d1"][idx] = np.minimum(first, distance)
    raw[side + "_d2"][idx] = np.minimum(second, np.maximum(first, distance))
    # A bounded time-to-arrival proxy. Speed alone does not include direction, hence the floor.
    arrival = distance / np.maximum(speed, 1.5)
    raw[side + "_arrival"][idx] = np.minimum(raw[side + "_arrival"][idx], arrival)
    for radius in (1, 3, 5):
        key = "{}_count{}".format(side, radius)
        raw[key][idx] += (distance <= radius).astype(np.int8)


def _concatenate_raw(parts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = set(parts[0])
    if any(set(part) != keys for part in parts):
        raise ValueError("raw section schemas differ")
    return {key: np.concatenate([part[key] for part in parts]) for key in sorted(keys)}


def cache_dfl(extra_dir: Path, cache_dir: Path) -> List[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source in sorted(extra_dir.glob(DFL_GLOB)):
        match_id = _match_id(source)
        target = cache_dir / (match_id + ".npz")
        outputs.append(target)
        signature = "{}:{}:{}".format(
            DFL_CACHE_SCHEMA, source.stat().st_size, source.stat().st_mtime_ns)
        if target.is_file():
            with np.load(target, allow_pickle=False) as cached:
                if str(cached["signature"].item()) == signature:
                    continue
        print("extracting DFL {}".format(match_id), flush=True)
        raw = extract_dfl_match(source)
        np.savez_compressed(target, signature=np.array(signature), **raw)
    if not outputs:
        raise FileNotFoundError("no {} below {}".format(DFL_GLOB, extra_dir))
    return outputs


def load_raw(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key != "signature"}


def extract_target_raw(data_dir: Path) -> Dict[str, np.ndarray]:
    """Extract the same provider-invariant geometry from the challenge ATD."""
    raw_path = data_dir / "tactical" / "BAR-ATH_tactical_atd.txt"
    metadata = resolve_tactical_metadata(data_dir)
    teams = team_of_tracks(metadata)
    n = GAMES[GAME]["n_video_frames"]
    length, width = GAMES[GAME]["pitch"]
    raw = _empty_raw(n)
    raw["pitch_length"][:] = length
    raw["pitch_width"][:] = width
    # Height is unavailable in the challenge ATD. Keeping NaN makes the domain difference
    # explicit to the estimator instead of falsely asserting that every ball is on the ground.
    raw["ball_z"][:] = np.nan
    for line in raw_path.open():
        fields = line.rstrip().split(":")
        if len(fields) != 3:
            continue
        frame = int(fields[0]) - 1
        ball = fields[2].split(",")
        if frame < 0 or frame >= n or not ball or ball[0].lower() == "nan":
            continue
        bx, by = float(ball[0]), float(ball[1])
        raw["visible"][frame] = True
        raw["ball_x"][frame] = (bx - 0.5) * length
        raw["ball_y"][frame] = (by - 0.5) * width
        if len(ball) > 2 and ball[2].lower() != "nan":
            raw["ball_speed"][frame] = float(ball[2])
        for slot, group in enumerate(fields[1].split(";")):
            if slot >= len(teams):
                break
            values = group.split(",")
            if not values or values[0].lower() == "nan":
                continue
            px, py = float(values[0]), float(values[1])
            distance = math.hypot((px - bx) * length, (py - by) * width)
            speed = float(values[2]) if len(values) > 2 and values[2].lower() != "nan" else 1.5
            side = "home" if int(teams[slot]) == H else "away"
            idx = np.array([frame], dtype=np.int64)
            _update_team_geometry(raw, side, idx, np.array([distance], dtype=np.float32),
                                  np.array([speed], dtype=np.float32))
    raw["labels"] = np.full(n, -1, dtype=np.int8)
    raw["run"] = np.full(n, -1, dtype=np.int16)
    for run_id, (_, lo, hi) in enumerate(half_video_blocks(GAME)):
        raw["run"][lo:hi + 1] = run_id
    return raw


def _finite(values: np.ndarray, cap: float) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = np.nan
    return np.clip(out, -cap, cap)


def _rolling_mean(values: np.ndarray, window: int, run: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    for run_id in np.unique(run[run >= 0]):
        idx = np.flatnonzero(run == run_id)
        x = values[idx].astype(np.float64)
        valid = np.isfinite(x)
        sums = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
        counts = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
        starts = np.maximum(0, np.arange(len(idx)) - window + 1)
        total = sums[np.arange(1, len(idx) + 1)] - sums[starts]
        count = counts[np.arange(1, len(idx) + 1)] - counts[starts]
        out[idx] = np.divide(total, count, out=np.full(len(idx), np.nan), where=count > 0)
    return out


def _future_mean(values: np.ndarray, window: int, run: np.ndarray) -> np.ndarray:
    return _rolling_mean(values[::-1], window, _reverse_run(run))[::-1]


def _reverse_run(run: np.ndarray) -> np.ndarray:
    rev = run[::-1].copy()
    ids = [int(x) for x in np.unique(rev) if x >= 0]
    mapping = {old: new for new, old in enumerate(ids)}
    for old, new in mapping.items():
        rev[rev == old] = new
    return rev


FEATURE_NAMES = (
    "visible", "abs_x", "abs_y", "boundary_distance", "ball_speed", "ball_acceleration",
    "home_d1", "away_d1", "home_d2", "away_d2", "distance_margin",
    "home_arrival", "away_arrival", "arrival_margin",
    "home_count1", "away_count1", "home_count3", "away_count3",
    "home_count5", "away_count5", "margin_mean_5", "margin_mean_15", "margin_mean_50",
    "margin_delta_15", "close_mean_15", "speed_mean_15", "height", "height_missing",
)


def make_features(raw: Dict[str, np.ndarray]) -> np.ndarray:
    visible = raw["visible"].astype(np.float32)
    length = raw["pitch_length"]
    width = raw["pitch_width"]
    x, y = raw["ball_x"], raw["ball_y"]
    abs_x = np.abs(x) / (length / 2.0)
    abs_y = np.abs(y) / (width / 2.0)
    boundary = np.minimum(length / 2.0 - np.abs(x), width / 2.0 - np.abs(y))
    speed = _finite(raw["ball_speed"], 40.0)
    acc = np.full(len(speed), np.nan, dtype=np.float32)
    same = np.r_[False, raw["run"][1:] == raw["run"][:-1]]
    acc[same] = np.abs(speed[same] - speed[np.flatnonzero(same) - 1]) * FPS
    hd1, ad1 = _finite(raw["home_d1"], 30), _finite(raw["away_d1"], 30)
    hd2, ad2 = _finite(raw["home_d2"], 30), _finite(raw["away_d2"], 30)
    margin = ad1 - hd1
    ha = _finite(raw["home_arrival"], 20)
    aa = _finite(raw["away_arrival"], 20)
    arrival_margin = aa - ha
    close = np.minimum(hd1, ad1)
    past15 = _rolling_mean(margin, 15, raw["run"])
    future15 = _future_mean(margin, 15, raw["run"])
    columns = [
        visible, abs_x, abs_y, boundary, speed, acc,
        hd1, ad1, hd2, ad2, margin, ha, aa, arrival_margin,
        raw["home_count1"], raw["away_count1"], raw["home_count3"], raw["away_count3"],
        raw["home_count5"], raw["away_count5"],
        _rolling_mean(margin, 5, raw["run"]), past15,
        _rolling_mean(margin, 50, raw["run"]), future15 - past15,
        _rolling_mean(close, 15, raw["run"]), _rolling_mean(speed, 15, raw["run"]),
        _finite(raw["ball_z"], 20), (~np.isfinite(raw["ball_z"])).astype(np.float32),
    ]
    return np.column_stack(columns).astype(np.float32)


def _mask_features(features: np.ndarray, rng: np.random.RandomState,
                   fraction: float) -> np.ndarray:
    out = features.copy()
    missing = rng.random_sample(len(out)) < fraction
    # Columns relying directly on ball location/control geometry. Temporal summaries remain as
    # the evidence available before/after a gap, mirroring the target pipeline.
    direct = list(range(1, 20)) + [24, 25, 26]
    out[np.ix_(missing, np.asarray(direct, dtype=np.int64))] = np.nan
    out[missing, 0] = 0.0
    return out


def training_indices(labels: np.ndarray, stride: int, rng: np.random.RandomState) -> np.ndarray:
    base = np.arange(0, len(labels), stride, dtype=np.int64)
    changes = np.flatnonzero(np.r_[False, labels[1:] != labels[:-1]])
    around = np.concatenate([changes + offset for offset in range(-25, 26)]) if len(changes) else []
    around = np.asarray(around, dtype=np.int64)
    around = around[(around >= 0) & (around < len(labels))]
    return np.unique(np.concatenate((base, around)))


class EmissionModel:
    def __init__(self, seed: int = CONFIG.random_seed):
        kwargs = dict(max_iter=120, learning_rate=0.08, max_leaf_nodes=15,
                      min_samples_leaf=80, l2_regularization=1.0, random_state=seed,
                      class_weight="balanced")
        self.dead = HistGradientBoostingClassifier(**kwargs)
        self.team = HistGradientBoostingClassifier(**kwargs)

    def fit(self, matches: Sequence[Tuple[np.ndarray, np.ndarray]],
            independent_mask: bool = True) -> "EmissionModel":
        rng = np.random.RandomState(CONFIG.random_seed)
        xs, ys = [], []
        for x, y in matches:
            idx = training_indices(y, CONFIG.sample_stride, rng)
            clean = x[idx]
            xs.append(clean)
            ys.append(y[idx])
            if independent_mask:
                xs.append(_mask_features(clean, rng, CONFIG.missing_fraction))
                ys.append(y[idx])
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        self.dead.fit(x, y == D)
        alive = y != D
        self.team.fit(x[alive], y[alive] == A)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p_dead = self.dead.predict_proba(x)[:, 1]
        p_away = self.team.predict_proba(x)[:, 1]
        probs = np.column_stack(((1 - p_dead) * (1 - p_away),
                                 (1 - p_dead) * p_away, p_dead))
        return np.clip(probs, 1e-5, 1.0)


class BoundaryModel:
    def __init__(self, seed: int = CONFIG.random_seed):
        self.model = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=1.0, random_state=seed, class_weight="balanced")

    def fit(self, matches: Sequence[Tuple[np.ndarray, np.ndarray]]) -> "BoundaryModel":
        rng = np.random.RandomState(CONFIG.random_seed + 1)
        xs, ys = [], []
        for x, labels in matches:
            changes = np.flatnonzero(np.r_[False, labels[1:] != labels[:-1]])
            positive = np.unique(np.concatenate([changes + o for o in range(-2, 3)]))
            positive = positive[(positive >= 0) & (positive < len(labels))]
            negative_pool = np.setdiff1d(np.arange(len(labels)), positive, assume_unique=False)
            count = min(len(negative_pool), max(1, 12 * len(positive)))
            negative = rng.choice(negative_pool, count, replace=False)
            idx = np.concatenate((positive, negative))
            xs.append(x[idx])
            ys.append(np.isin(idx, positive))
        self.model.fit(np.concatenate(xs), np.concatenate(ys))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


OFFENSIVE_CODES = {
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD",
    "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL", "ATTACKING TRANSITION",
    "SET PIECES", "GOALS",
}
DEFENSIVE_CODES = {
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION",
}


def smart_tag_features(data_dir: Path, n_frames: int) -> Dict[str, np.ndarray]:
    votes = np.zeros((n_frames, 2), dtype=np.float32)
    set_piece = np.zeros(n_frames, dtype=np.float32)
    boundary = np.zeros(n_frames, dtype=np.float32)
    transition = np.zeros(n_frames, dtype=np.float32)
    events = load_smart_tagging(data_dir=str(data_dir))
    for event in events:
        lo = max(0, event["start_frame"])
        hi = min(n_frames, event["end_frame"])
        if hi <= lo:
            continue
        code = (event["code"] or "").strip().upper()
        team_text = (event["tags"].get("Team") or "").strip().lower()
        team = H if "barcelona" in team_text else (A if "atletic" in team_text else None)
        if team is not None and code in OFFENSIVE_CODES | DEFENSIVE_CODES:
            possessor = 1 - team if code in DEFENSIVE_CODES else team
            votes[lo:hi, possessor] += 1.0
        if code == "SET PIECES":
            set_piece[lo:hi] = 1.0
        if code in ("ATTACKING TRANSITION", "DEFENSIVE TRANSITION"):
            transition[lo:hi] = 1.0
        # Tag boundaries are only coarse evidence. Spread a triangular vote over one second.
        for edge in (lo, hi):
            left, right = max(0, edge - FPS), min(n_frames, edge + FPS + 1)
            distance = np.abs(np.arange(left, right) - edge) / float(FPS)
            boundary[left:right] = np.maximum(boundary[left:right], 1.0 - distance)
    total = votes.sum(axis=1)
    signed = np.divide(votes[:, H] - votes[:, A], total,
                       out=np.zeros(n_frames, dtype=np.float32), where=total > 0)
    confidence = np.divide(np.abs(votes[:, H] - votes[:, A]), total,
                           out=np.zeros(n_frames, dtype=np.float32), where=total > 0)
    return {"signed_team": signed, "confidence": confidence, "set_piece": set_piece,
            "boundary": boundary, "transition": transition}


def apply_smart_tags(probs: np.ndarray, tags: Dict[str, np.ndarray]) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-6, 1.0))
    signed = tags["signed_team"] * tags["confidence"]
    logp[:, H] += CONFIG.tag_team_weight * signed
    logp[:, A] -= CONFIG.tag_team_weight * signed
    logp[:, D] += CONFIG.tag_dead_weight * tags["set_piece"]
    return _softmax(logp)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def viterbi_decode(probs: np.ndarray, runs: Sequence[Tuple[int, int]],
                   boundary: np.ndarray | None = None,
                   switch_cost: float = CONFIG.viterbi_switch_cost) -> np.ndarray:
    out = np.full(len(probs), D, dtype=np.int8)
    logp = np.log(np.clip(probs, 1e-8, 1.0))
    for lo, hi in runs:
        length = hi - lo
        back = np.zeros((length, 3), dtype=np.int8)
        score = logp[lo].copy()
        for offset, frame in enumerate(range(lo + 1, hi), start=1):
            discount = 0.0 if boundary is None else CONFIG.boundary_switch_discount * boundary[frame]
            cost = max(0.15, switch_cost - discount)
            transition = np.full((3, 3), -cost, dtype=np.float64)
            np.fill_diagonal(transition, 0.0)
            candidates = score[:, None] + transition
            back[offset] = np.argmax(candidates, axis=0)
            score = logp[frame] + np.max(candidates, axis=0)
        states = np.empty(length, dtype=np.int8)
        states[-1] = int(np.argmax(score))
        for offset in range(length - 1, 0, -1):
            states[offset - 1] = back[offset, states[offset]]
        out[lo:hi] = states
    return merge_micro_runs(out, probs, runs)


def _segments(labels: np.ndarray, lo: int, hi: int) -> List[Tuple[int, int, int]]:
    if hi <= lo:
        return []
    edges = np.flatnonzero(labels[lo + 1:hi] != labels[lo:hi - 1]) + lo + 1
    bounds = np.r_[lo, edges, hi]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def merge_micro_runs(labels: np.ndarray, probs: np.ndarray,
                     runs: Sequence[Tuple[int, int]]) -> np.ndarray:
    out = labels.copy()
    for lo, hi in runs:
        changed = True
        while changed:
            changed = False
            segments = _segments(out, lo, hi)
            for index, (a, b, state) in enumerate(segments):
                minimum = CONFIG.min_dead_run if state == D else CONFIG.min_team_run
                if b - a >= minimum or len(segments) == 1:
                    continue
                candidates = []
                if index > 0:
                    candidates.append(segments[index - 1][2])
                if index + 1 < len(segments):
                    candidates.append(segments[index + 1][2])
                replacement = max(set(candidates), key=lambda k: float(np.log(probs[a:b, k]).sum()))
                out[a:b] = replacement
                changed = True
                break
    return out


def run_ranges(run: np.ndarray) -> List[Tuple[int, int]]:
    ranges = []
    for run_id in np.unique(run[run >= 0]):
        idx = np.flatnonzero(run == run_id)
        ranges.append((int(idx[0]), int(idx[-1] + 1)))
    return ranges


def hard_labels(probs: np.ndarray, run: np.ndarray) -> np.ndarray:
    out = np.argmax(probs, axis=1).astype(np.int8)
    out[run < 0] = D
    return out


def missing_gap_lengths(visible: np.ndarray, run: np.ndarray) -> np.ndarray:
    """Lengths of missing-ball gaps inside valid runs; match breaks are excluded."""
    lengths = []
    for lo, hi in run_ranges(run):
        missing = ~visible[lo:hi]
        edges = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        lengths.extend((stops - starts).tolist())
    return np.asarray(lengths or [1], dtype=np.int32)


def mask_raw_contiguous(raw: Dict[str, np.ndarray], gap_lengths: np.ndarray,
                        seed: int, fraction: float = CONFIG.missing_fraction) -> Dict[str, np.ndarray]:
    """Apply target-shaped contiguous ball gaps to a labelled DFL match."""
    rng = np.random.RandomState(seed)
    out = {key: value.copy() for key, value in raw.items()}
    mask = np.zeros(len(raw["visible"]), dtype=bool)
    target = int(round(fraction * np.sum(raw["run"] >= 0)))
    attempts = 0
    ranges = run_ranges(raw["run"])
    while int(mask.sum()) < target and attempts < target * 2:
        lo, hi = ranges[rng.randint(len(ranges))]
        length = int(gap_lengths[rng.randint(len(gap_lengths))])
        length = max(1, min(length, hi - lo))
        start = rng.randint(lo, hi - length + 1)
        mask[start:start + length] = True
        attempts += 1
    out["visible"][mask] = False
    for key in ("ball_x", "ball_y", "ball_z", "ball_speed"):
        out[key][mask] = np.nan
    for side in ("home", "away"):
        for suffix in ("d1", "d2", "arrival"):
            out[side + "_" + suffix][mask] = np.inf
        for radius in (1, 3, 5):
            out["{}_count{}".format(side, radius)][mask] = 0
    return out


CONTEXT_COLUMNS = (0, 4, 6, 7, 10, 13, 23, 24)
CONTEXT_OFFSETS = (-30, -10, 10, 30)


def make_context_features(features: np.ndarray, run: np.ndarray) -> np.ndarray:
    """Add sparse bidirectional temporal context without assuming persistent player IDs."""
    additions = []
    n = len(features)
    for offset in CONTEXT_OFFSETS:
        shifted = np.full((n, len(CONTEXT_COLUMNS)), np.nan, dtype=np.float32)
        if offset < 0:
            source = np.arange(-offset, n)
            target = source + offset
        else:
            source = np.arange(0, n - offset)
            target = source + offset
        same = (run[source] >= 0) & (run[source] == run[target])
        shifted[target[same]] = features[source[same]][:, CONTEXT_COLUMNS]
        additions.append(shifted)

    # Frames since and until a visible observation explicitly describe occlusion gaps.
    since = np.full(n, np.nan, dtype=np.float32)
    until = np.full(n, np.nan, dtype=np.float32)
    for lo, hi in run_ranges(run):
        last = -1
        for index in range(lo, hi):
            if features[index, 0] > 0.5:
                last = index
            since[index] = index - last if last >= 0 else np.nan
        nxt = -1
        for index in range(hi - 1, lo - 1, -1):
            if features[index, 0] > 0.5:
                nxt = index
            until[index] = nxt - index if nxt >= 0 else np.nan
    return np.column_stack([features] + additions + [since, until]).astype(np.float32)


def hierarchical_decode(probs: np.ndarray, runs: Sequence[Tuple[int, int]],
                        dead_bias: float = -0.75, dead_switch_cost: float = 6.0,
                        team_switch_cost: float = 8.0) -> np.ndarray:
    """Decode ball status first, then team possession inside each live interval."""
    out = np.full(len(probs), D, dtype=np.int8)
    eps = 1e-8
    dead_prob = np.clip(probs[:, D], eps, 1 - eps)
    dead_logits = np.log(dead_prob / (1 - dead_prob)) + dead_bias
    for lo, hi in runs:
        # Two-state Viterbi: 0 alive, 1 dead.
        emissions = np.column_stack((np.zeros(hi - lo), dead_logits[lo:hi]))
        back = np.zeros((hi - lo, 2), dtype=np.int8)
        score = emissions[0].copy()
        transition = np.array([[0.0, -dead_switch_cost],
                               [-dead_switch_cost, 0.0]])
        for offset in range(1, hi - lo):
            candidates = score[:, None] + transition
            back[offset] = np.argmax(candidates, axis=0)
            score = emissions[offset] + np.max(candidates, axis=0)
        status = np.empty(hi - lo, dtype=np.int8)
        status[-1] = int(np.argmax(score))
        for offset in range(hi - lo - 1, 0, -1):
            status[offset - 1] = back[offset, status[offset]]

        alive = status == 0
        edges = np.diff(np.r_[False, alive, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            a, b = lo + int(start), lo + int(stop)
            # Copy: normalising a view here used to mutate the caller's emission matrix and made
            # later decoders depend silently on execution order.
            team_probs = probs[a:b, :2].copy()
            team_probs /= np.maximum(team_probs.sum(axis=1, keepdims=True), eps)
            team_logp = np.log(np.clip(team_probs, eps, 1.0))
            team_back = np.zeros((b - a, 2), dtype=np.int8)
            team_score = team_logp[0].copy()
            team_transition = np.array([[0.0, -team_switch_cost],
                                        [-team_switch_cost, 0.0]])
            for offset in range(1, b - a):
                candidates = team_score[:, None] + team_transition
                team_back[offset] = np.argmax(candidates, axis=0)
                team_score = team_logp[offset] + np.max(candidates, axis=0)
            states = np.empty(b - a, dtype=np.int8)
            states[-1] = int(np.argmax(team_score))
            for offset in range(b - a - 1, 0, -1):
                states[offset - 1] = team_back[offset, states[offset]]
            out[a:b] = states
    return merge_micro_runs(out, probs, runs)


def hierarchical_alive_calibrate(probs: np.ndarray,
                                 runs: Sequence[Tuple[int, int]],
                                 dead_bias: float = 0.0,
                                 dead_switch_cost: float = 8.0) -> np.ndarray:
    """Explicitly reproduce the useful alive-state calibration formerly caused by mutation.

    The hierarchical status path identifies temporally coherent live intervals. Within those
    intervals, H/A emissions are conditioned on the ball being alive while the dead emission is
    retained as a competing hypothesis for the downstream sparse decoder.
    """
    status_path = hierarchical_decode(
        probs.copy(), runs, dead_bias=dead_bias, dead_switch_cost=dead_switch_cost)
    out = probs.copy()
    alive = status_path != D
    team_total = out[:, :2].sum(axis=1)
    valid = alive & (team_total > 1e-8)
    out[valid, :2] /= team_total[valid, None]
    return out


def sparse_boundary_decode(probs: np.ndarray, runs: Sequence[Tuple[int, int]],
                           boundary: np.ndarray, threshold: float = 0.6,
                           candidate_cost: float = 5.0, other_cost: float = 14.0) -> np.ndarray:
    """Allow cheap switches only near a high-confidence learned boundary candidate."""
    gate = (boundary >= threshold).astype(np.float32)
    # viterbi_decode supports one base cost and a fixed discount multiplier. Rescale the gate so
    # non-candidates cost other_cost and candidates cost candidate_cost.
    old_discount = CONFIG.boundary_switch_discount
    effective = gate * ((other_cost - candidate_cost) / old_discount)
    return viterbi_decode(probs, runs, boundary=effective, switch_cost=other_cost)


def _sparse_boundary_peaks(boundary: np.ndarray, runs: Sequence[Tuple[int, int]],
                           threshold: float, smooth_radius: int,
                           gate_radius: int, min_separation: int) -> np.ndarray:
    """Turn a dense boundary score into isolated, locally adjustable switch windows."""
    gate = np.zeros(len(boundary), dtype=np.float32)
    for lo, hi in runs:
        scores = np.asarray(boundary[lo:hi], dtype=np.float64)
        if smooth_radius:
            width = 2 * smooth_radius + 1
            scores = np.convolve(scores, np.ones(width) / width, mode="same")
        local_max = np.r_[True, scores[1:] >= scores[:-1]] & np.r_[scores[:-1] > scores[1:], True]
        candidates = np.flatnonzero(local_max & (scores >= threshold))
        # Greedy temporal non-maximum suppression retains the strongest point on a plateau and
        # prevents a burst of correlated boundary scores from licensing several possessions.
        selected = []
        for offset in candidates[np.argsort(scores[candidates])[::-1]]:
            if all(abs(int(offset) - previous) >= min_separation for previous in selected):
                selected.append(int(offset))
        for offset in selected:
            centre = lo + offset
            gate[max(lo, centre - gate_radius):min(hi, centre + gate_radius + 1)] = 1.0
    return gate


def merge_short_runs(labels: np.ndarray, probs: np.ndarray,
                     runs: Sequence[Tuple[int, int]], team_min: int,
                     dead_min: int) -> np.ndarray:
    """Approximate a semi-Markov duration constraint by merging weak short runs."""
    out = labels.copy()
    for lo, hi in runs:
        changed = True
        while changed:
            changed = False
            segments = _segments(out, lo, hi)
            for index, (start, stop, state) in enumerate(segments):
                minimum = dead_min if state == D else team_min
                if minimum <= 0 or stop - start >= minimum or len(segments) == 1:
                    continue
                neighbours = []
                if index:
                    neighbours.append(segments[index - 1][2])
                if index + 1 < len(segments):
                    neighbours.append(segments[index + 1][2])
                replacement = max(
                    set(neighbours),
                    key=lambda candidate: float(np.log(probs[start:stop, candidate]).sum()),
                )
                out[start:stop] = replacement
                changed = True
                break
    return merge_micro_runs(out, probs, runs)


def sparse_peak_boundary_decode(
        probs: np.ndarray, runs: Sequence[Tuple[int, int]], boundary: np.ndarray,
        threshold: float = 0.7, candidate_cost: float = 5.0,
        other_cost: float = 14.0, smooth_radius: int = 2,
        gate_radius: int = 4, min_separation: int = 12,
        team_min: int = 10, dead_min: int = 8) -> np.ndarray:
    """Sparse-boundary Viterbi with peak suppression and explicit duration regularisation."""
    gate = _sparse_boundary_peaks(
        boundary, runs, threshold, smooth_radius, gate_radius, min_separation)
    effective = gate * ((other_cost - candidate_cost) / CONFIG.boundary_switch_discount)
    decoded = viterbi_decode(probs, runs, boundary=effective, switch_cost=other_cost)
    return merge_short_runs(decoded, probs, runs, team_min, dead_min)


def boundary_veto_decode(probs: np.ndarray, runs: Sequence[Tuple[int, int]],
                         boundary: np.ndarray, threshold: float = 0.5,
                         radius: int = 8) -> np.ndarray:
    """Merge predicted runs whose boundary lacks local corroboration."""
    out = viterbi_decode(probs, runs)
    for lo, hi in runs:
        changed = True
        while changed:
            changed = False
            segments = _segments(out, lo, hi)
            for index in range(1, len(segments)):
                edge = segments[index][0]
                local = boundary[max(lo, edge - radius):min(hi, edge + radius + 1)]
                if len(local) and float(np.max(local)) >= threshold:
                    continue
                left, right = segments[index - 1], segments[index]
                # Remove the weaker of the two adjacent runs using calibrated emission support.
                left_support = float(np.log(probs[left[0]:left[1], left[2]]).mean())
                right_support = float(np.log(probs[right[0]:right[1], right[2]]).mean())
                if left_support >= right_support:
                    out[right[0]:right[1]] = left[2]
                else:
                    out[left[0]:left[1]] = right[2]
                changed = True
                break
    return merge_micro_runs(out, probs, runs)


def frame_vote(predictions: Sequence[np.ndarray], probs: np.ndarray) -> np.ndarray:
    votes = np.zeros((len(probs), 3), dtype=np.int16)
    for pred in predictions:
        votes[np.arange(len(pred)), pred] += 1
    best = votes.max(axis=1, keepdims=True)
    tied = votes == best
    return np.argmax(np.where(tied, probs, -1.0), axis=1).astype(np.int8)


def spell_vote(predictions: Sequence[np.ndarray], probs: np.ndarray, run: np.ndarray,
               learned_boundary: np.ndarray) -> np.ndarray:
    radius = CONFIG.vote_tolerance
    vote = np.zeros(len(probs), dtype=np.float32)
    for pred in predictions:
        changes = np.flatnonzero(np.r_[False, pred[1:] != pred[:-1]])
        marks = np.zeros(len(pred), dtype=np.float32)
        marks[changes] = 1.0
        if radius:
            marks = np.convolve(marks, np.ones(2 * radius + 1, dtype=np.float32), mode="same")
            marks = np.clip(marks, 0, 1)
        vote += marks
    vote /= max(1, len(predictions))
    boundary = 0.55 * learned_boundary + 0.45 * vote
    # Hard labels become weak spell-level experts; retain the calibrated ML probabilities.
    pooled = 0.50 * probs
    for pred in predictions:
        onehot = np.eye(3, dtype=np.float32)[pred]
        pooled += (0.50 / len(predictions)) * (0.85 * onehot + 0.15 / 3.0)
    pooled /= pooled.sum(axis=1, keepdims=True)
    return viterbi_decode(pooled, run_ranges(run), boundary=boundary)


def codes_to_labels(codes: np.ndarray, run: np.ndarray) -> np.ndarray:
    out = np.full(len(codes), "X", dtype="<U1")
    valid = run >= 0
    out[valid] = LABELS[codes[valid]]
    return out


def save_config(path: Path, cache_paths: Sequence[Path]) -> None:
    payload = {
        "experiment_config": CONFIG.__dict__,
        "feature_names": list(FEATURE_NAMES),
        "dfl_matches": [p.stem for p in cache_paths],
        "sklearn_model": "HistGradientBoostingClassifier",
        "public_ground_truth_used_for_fitting": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
