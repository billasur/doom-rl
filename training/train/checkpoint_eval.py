"""Fixed development maps for selecting weights during a burst."""

import hashlib
import json
from pathlib import Path
from statistics import mean

import torch

from eval.eval_suite import SCORE, run_episode
from maps.gen_wads import TIERS, textmap, write_wad
from maps.micro_drills import lock_drill
from models.v1_spatial_net import SpatialActorCritic, pack_telemetry
from submission.agent import DoomAgent

MAP_SEEDS = (71001, 71101, 71201, 71301)
EPISODE_SEEDS = (81001, 81101, 81201, 81301)
SPECIALIST_TIERS = {"keycard": (2,), "micro_keycard": (2,), "explore": (1,), "combat": (0, 3), "mixed": (0, 1, 2, 3)}


def specialist_score(role, row):
    if role in ("keycard", "micro_keycard"):
        return (2 * row["keys_picked"] + 3 * bool(row["milestones"]["branch_return"])
                + 6 * row["locked_opened"] + 10 * row["exit"])
    if role == "explore":
        return 10 * row["exit"] + 2 * row["doors_opened"] + 0.025 * min(row["cells_visited"], 128)
    if role == "combat":
        opening = row["combat"]["first_room_clear_step"] is not None
        return (4 * opening + 2 * row["kills"] + 4 * row["exit"] - 2 * row["dead"]
                - 0.005 * row["damage_taken"])
    return row["score"]


class CheckpointAgent(DoomAgent):
    def __init__(self, observation_space, action_space, **features):
        self.device = torch.device("cpu")
        self.spatial = True
        self.depth = features.get("depth", False)
        self.temperature = 1.0
        self.telemetry = pack_telemetry
        self.model = SpatialActorCritic(observation_space, action_space, **features).eval()
        self.reset()


class CheckpointEval:
    def __init__(self, spec, workdir: Path, ticks, *, features=None, tier_weights=None, role="mixed"):
        workdir.mkdir(exist_ok=True)
        self.ticks = ticks
        self.role = role
        self.weights = tier_weights or [1] * len(TIERS)
        self.jobs = []
        self.reverse_starts = []
        hashes = []
        for index in SPECIALIST_TIERS[role]:
            for repeat in range(6 if role == "micro_keycard" else 3 if role == "keycard" else 2):
                map_seed, seed = MAP_SEEDS[index] + repeat, EPISODE_SEEDS[index] + repeat
                stage = ("micro_lock" if repeat < 3 else "branch") if role == "micro_keycard" else None
                wad = workdir / f"seed_{map_seed}.wad"
                write_wad(wad, lock_drill(map_seed) if stage == "micro_lock" else textmap(index, map_seed, reverse_stage=stage))
                self.jobs.append((wad, "MAP01", TIERS[index], seed))
                self.reverse_starts.append(stage)
                hashes.append(hashlib.sha256(wad.read_bytes()).hexdigest())
        self.protocol = {"version": 3, "role": role, "ticks": ticks, "seeds": [job[3] for job in self.jobs],
                         "wads": hashes, "score_weights": SCORE, "tier_weights": self.weights}
        with torch.random.fork_rng(devices=[]):
            self.agent = CheckpointAgent(*spec, **(features or {}))

    def evaluate(self, weights):
        self.agent.model.load_state_dict(weights)
        with torch.random.fork_rng(devices=[]):
            scores = {}
            for job, stage in zip(self.jobs, self.reverse_starts):
                row = run_episode(self.agent, *job, self.ticks, reverse_stage=stage,
                                  log_progression=self.role in ("keycard", "micro_keycard", "combat"))[0]
                scores.setdefault(job[2], []).append(specialist_score(self.role, row))
        scores = {tier: mean(values) for tier, values in scores.items()}
        weights = {tier: self.weights[TIERS.index(tier)] for tier in scores}
        denominator = sum(weights.values())
        if not denominator:
            raise ValueError(f"Checkpoint weights exclude all {self.role} evaluation tiers")
        return round(sum(score * weights[tier] for tier, score in scores.items()) / denominator, 4), scores
