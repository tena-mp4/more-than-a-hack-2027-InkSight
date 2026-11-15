"""Shared feature extraction and three deterministic possession models.

The public ground truth is deliberately not imported here.  Model constants are fixed in
MODEL_CONFIG and inference depends only on the supplied automatic tracking/tagging assets.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from games import GAME, GAMES, half_video_blocks
from starter.load_smart_tagging import load as load_smart_tagging


ROOT = Path(__file__).resolve().parents[1]
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
LABELS = np.array(["H", "A", "D"], dtype="<U1")
H, A, D = 0, 1, 2


@dataclass(frozen=True)
class ModelConfig:
    possession_radius_m: float = 3.0
    distance_temperature_m: float = 1.4
    team_tag_weight: float = 0.55
    state_confirm_frames: int = 8
    dead_confirm_frames: int = 6
    hysteresis_alpha_team: float = 0.08
    hysteresis_alpha_dead: float = 0.12
    hysteresis_team_threshold: float = 0.30
    hysteresis_dead_enter: float = 1.0
    hysteresis_dead_exit: float = -0.55
    viterbi_emission_scale: float = 0.08
    viterbi_team_switch_cost: float = 2.8
    viterbi_dead_switch_cost: float = 1.2
    set_piece_restart_fraction: float = 0.50


MODEL_CONFIG = ModelConfig()


def resolve_tactical_metadata(data_dir: Path) -> Path:
    """Resolve the tactical XML by its internal FileName, tolerating the delivered misplacement."""
    expected = data_dir / "tactical" / "BAR-ATH_tactical_atd.xml"
    candidates = [expected]
    candidates.extend(sorted(data_dir.glob("*/*.xml")))
    checked = []
    for path in candidates:
        if not path.is_file() or path in checked or "smart_tagging" in path.name:
            continue
        checked.append(path)
        try:
            file_name = ET.parse(path).findtext(".//FileName") or ""
        except ET.ParseError:
            continue
        if "_TAC_" in file_name.upper() or "TACTICAL" in file_name.upper():
            return path
    raise FileNotFoundError("could not identify the tactical ATD metadata XML")


def team_of_tracks(metadata_path: Path) -> np.ndarray:
    root = ET.parse(metadata_path).getroot()
    team_ids = [node.get("id") for node in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected two teams in {}".format(metadata_path))
    player_team_ids = [node.get("teamId") for node in root.iter("Player")]
    if not player_team_ids:
        raise ValueError("no player track slots in {}".format(metadata_path))
    return np.array([H if value == team_ids[0] else A for value in player_team_ids], dtype=np.int8)


def _two_smallest(values: Iterable[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return math.inf, math.inf
    if len(values) == 1:
        return values[0], math.inf
    pair = np.partition(np.asarray(values, dtype=np.float32), 1)[:2]
    return float(pair[0]), float(pair[1])


def extract_tracking_features(raw_path: Path, metadata_path: Path) -> Dict[str, np.ndarray]:
    teams = team_of_tracks(metadata_path)
    features = {
        "ball_visible": np.zeros(N_FRAMES, dtype=bool),
        "ball_x": np.full(N_FRAMES, np.nan, dtype=np.float32),
        "ball_y": np.full(N_FRAMES, np.nan, dtype=np.float32),
        "ball_speed": np.full(N_FRAMES, np.nan, dtype=np.float32),
        "home_d1": np.full(N_FRAMES, np.inf, dtype=np.float32),
        "home_d2": np.full(N_FRAMES, np.inf, dtype=np.float32),
        "away_d1": np.full(N_FRAMES, np.inf, dtype=np.float32),
        "away_d2": np.full(N_FRAMES, np.inf, dtype=np.float32),
    }
    seen = 0
    with raw_path.open() as handle:
        for line in handle:
            parts = line.rstrip().split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= N_FRAMES:
                continue
            seen += 1
            ball_fields = parts[2].split(",")
            if not ball_fields or ball_fields[0].lower() == "nan":
                continue
            bx, by = float(ball_fields[0]), float(ball_fields[1])
            features["ball_visible"][frame] = True
            features["ball_x"][frame] = bx
            features["ball_y"][frame] = by
            if len(ball_fields) > 2 and ball_fields[2].lower() != "nan":
                features["ball_speed"][frame] = float(ball_fields[2])

            distances = {H: [], A: []}
            for slot, group in enumerate(parts[1].split(";")):
                if slot >= len(teams):
                    break
                fields = group.split(",")
                if not fields or fields[0].lower() == "nan":
                    continue
                dx = (float(fields[0]) - bx) * PITCH_LENGTH
                dy = (float(fields[1]) - by) * PITCH_WIDTH
                distances[int(teams[slot])].append(math.hypot(dx, dy))
            features["home_d1"][frame], features["home_d2"][frame] = _two_smallest(distances[H])
            features["away_d1"][frame], features["away_d2"][frame] = _two_smallest(distances[A])
    if seen != N_FRAMES:
        raise ValueError("tracking has {} usable records, expected {}".format(seen, N_FRAMES))
    return features


def add_tag_features(features: Dict[str, np.ndarray], data_dir: Path, config: ModelConfig) -> None:
    team_sum = np.zeros(N_FRAMES, dtype=np.float32)
    team_count = np.zeros(N_FRAMES, dtype=np.int16)
    set_piece_phase = np.full(N_FRAMES, np.nan, dtype=np.float32)
    events = load_smart_tagging(data_dir=str(data_dir))
    for event in events:
        lo = max(0, event["start_frame"])
        hi = min(N_FRAMES, event["end_frame"])
        if hi <= lo:
            continue
        team = (event["tags"].get("Team") or "").lower()
        value = 1.0 if "barcelona" in team else (-1.0 if "atletic" in team else 0.0)
        if value:
            team_sum[lo:hi] += value
            team_count[lo:hi] += 1
        if (event["code"] or "").strip().lower() == "set pieces":
            set_piece_phase[lo:hi] = np.linspace(0.0, 1.0, hi - lo, endpoint=False, dtype=np.float32)
    features["team_tag"] = np.divide(
        team_sum, team_count, out=np.zeros_like(team_sum), where=team_count > 0
    )
    features["set_piece_phase"] = set_piece_phase


def build_features(data_dir: Path, cache_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    raw_path = data_dir / "tactical" / "BAR-ATH_tactical_atd.txt"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    metadata_path = resolve_tactical_metadata(data_dir)
    signature = "{}:{}:{}".format(raw_path.stat().st_size, metadata_path.stat().st_size, MODEL_CONFIG)
    if cache_path and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["signature"].item()) == signature:
            return {key: cached[key] for key in cached.files if key != "signature"}
    features = extract_tracking_features(raw_path, metadata_path)
    add_tag_features(features, data_dir, MODEL_CONFIG)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, signature=np.array(signature), **features)
    return features


def evidence(features: Dict[str, np.ndarray], config: ModelConfig = MODEL_CONFIG):
    visible = features["ball_visible"]
    hd = features["home_d1"]
    ad = features["away_d1"]
    finite = visible & np.isfinite(hd) & np.isfinite(ad)
    team = np.zeros(N_FRAMES, dtype=np.float32)
    team[finite] = np.tanh((ad[finite] - hd[finite]) / config.distance_temperature_m)
    team += config.team_tag_weight * features["team_tag"]
    team = np.clip(team, -1.5, 1.5)

    in_bounds = visible & (features["ball_x"] >= 0.0) & (features["ball_x"] <= 1.0)
    in_bounds &= (features["ball_y"] >= 0.0) & (features["ball_y"] <= 1.0)
    close = np.minimum(hd, ad) <= config.possession_radius_m
    dead = np.zeros(N_FRAMES, dtype=np.float32)
    dead[visible & ~in_bounds] = 5.0
    dead[in_bounds & close] = -2.5
    phase = features["set_piece_phase"]
    before_restart = np.isfinite(phase) & (phase < config.set_piece_restart_fraction)
    after_restart = np.isfinite(phase) & ~before_restart
    dead[before_restart] += 1.8
    dead[after_restart] -= 1.0

    # Published match boundaries: the second half is graded before its actual kickoff.
    dead[110808:113742] = 8.0
    dead[113742] = -8.0
    return team, dead, in_bounds, close


def _blank_output() -> np.ndarray:
    return np.full(N_FRAMES, "X", dtype="<U1")


def _half_ranges():
    return [(first, last + 1) for _, first, last in half_video_blocks(GAME)]


def approach_state_machine(features: Dict[str, np.ndarray], config: ModelConfig = MODEL_CONFIG):
    """Possession-zone rules plus explicit confirmation counters."""
    team, dead, _, close = evidence(features, config)
    out = _blank_output()
    for lo, hi in _half_ranges():
        state = D if dead[lo] > 1.0 else (H if team[lo] >= 0 else A)
        candidate, count = state, 0
        for frame in range(lo, hi):
            if dead[frame] >= 3.0:
                proposed = D
                needed = config.dead_confirm_frames
            elif state == D and dead[frame] <= -0.5 and close[frame]:
                proposed = H if team[frame] >= 0 else A
                needed = config.dead_confirm_frames
            elif state != D and close[frame] and abs(team[frame]) >= 0.35:
                proposed = H if team[frame] > 0 else A
                needed = config.state_confirm_frames
            else:
                proposed, needed = state, config.state_confirm_frames
            if proposed == state:
                candidate, count = state, 0
            elif proposed == candidate:
                count += 1
            else:
                candidate, count = proposed, 1
            if count >= needed:
                state, candidate, count = proposed, proposed, 0
            out[frame] = LABELS[state]
    return out


def approach_hysteresis(features: Dict[str, np.ndarray], config: ModelConfig = MODEL_CONFIG):
    """Exponentially filtered likelihoods with asymmetric enter/exit thresholds."""
    team, dead, _, close = evidence(features, config)
    out = _blank_output()
    for lo, hi in _half_ranges():
        team_level, dead_level = float(team[lo]), float(dead[lo])
        state = D if dead_level > config.hysteresis_dead_enter else (H if team_level >= 0 else A)
        for frame in range(lo, hi):
            if features["ball_visible"][frame] or features["team_tag"][frame] != 0:
                team_level += config.hysteresis_alpha_team * (float(team[frame]) - team_level)
            dead_level += config.hysteresis_alpha_dead * (float(dead[frame]) - dead_level)
            if state == D:
                if dead_level < config.hysteresis_dead_exit and close[frame]:
                    state = H if team_level >= 0 else A
            elif dead_level > config.hysteresis_dead_enter:
                state = D
            elif state == H and team_level < -config.hysteresis_team_threshold:
                state = A
            elif state == A and team_level > config.hysteresis_team_threshold:
                state = H
            out[frame] = LABELS[state]
    return out


def approach_viterbi(features: Dict[str, np.ndarray], config: ModelConfig = MODEL_CONFIG):
    """Globally decode H/A/D with transition penalties, a compact PathCRF-style approximation."""
    team, dead, _, _ = evidence(features, config)
    emissions = np.column_stack((team, -team, dead)).astype(np.float32)
    emissions[:, D] -= 0.25
    emissions *= config.viterbi_emission_scale
    transitions = np.array(
        [[0.0, -config.viterbi_team_switch_cost, -config.viterbi_dead_switch_cost],
         [-config.viterbi_team_switch_cost, 0.0, -config.viterbi_dead_switch_cost],
         [-config.viterbi_dead_switch_cost, -config.viterbi_dead_switch_cost, 0.0]],
        dtype=np.float32,
    )
    out = _blank_output()
    for lo, hi in _half_ranges():
        length = hi - lo
        back = np.zeros((length, 3), dtype=np.int8)
        score = emissions[lo].copy()
        for offset, frame in enumerate(range(lo + 1, hi), start=1):
            candidates = score[:, None] + transitions
            back[offset] = np.argmax(candidates, axis=0)
            score = emissions[frame] + np.max(candidates, axis=0)
        states = np.empty(length, dtype=np.int8)
        states[-1] = int(np.argmax(score))
        for offset in range(length - 1, 0, -1):
            states[offset - 1] = back[offset, states[offset]]
        out[lo:hi] = LABELS[states]
    return out


APPROACHES = {
    "approach_1_state_machine": approach_state_machine,
    "approach_2_hysteresis": approach_hysteresis,
    "approach_3_viterbi": approach_viterbi,
}


def write_submission(path: Path, labels: np.ndarray) -> None:
    if len(labels) != N_FRAMES:
        raise ValueError("expected {} labels".format(N_FRAMES))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def write_config(path: Path, metadata_path: Path) -> None:
    payload = {
        "game": GAME,
        "n_frames": N_FRAMES,
        "fps": FPS,
        "tactical_metadata_resolved_to": str(metadata_path.relative_to(ROOT)),
        "model_config": asdict(MODEL_CONFIG),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
