"""Evaluate a checkpoint on held-out WADs and plot progression by tier."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from doom_env import make_env
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from maps.gen_wads import TIERS
from maps.pool import FREEDOOM_EVAL, freedoom_wad, obsidian_maps
from eval.progression import MILESTONES, Progression
from submission.agent import V1_WEIGHTS_PATH, DoomAgent
from train.universal_reward import UniversalReward

SCORE = {"exit": 10.0, "kills": 1.0, "keys_picked": 2.0, "doors_opened": 0.5,
         "damage_taken": -0.01, "dead": -1.0}


def run_episode(agent, path, map_name, tier, seed, ticks, *, log_progression=True, reverse_stage=None):
    env = UniversalReward(make_env(path, map_name, episode_ticks=ticks, render_buffers=getattr(agent, "depth", False)),
                          reverse_stage=reverse_stage)
    try:
        torch.random.default_generator.manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        agent.reset()
        state = env.previous
        positions, latencies = [(state["x"], state["y"])], []
        progression = Progression(env, tier, state, env.last_objects[2]) if log_progression else None
        base_reward = 0.0
        while True:
            start = perf_counter()
            action = agent.act(obs)
            latencies.append((perf_counter() - start) * 1000)
            previous_obs = obs
            obs, _, terminated, truncated, info = env.step(action)
            base_reward += info["base_reward"]
            state = env.previous
            positions.append((state["x"], state["y"]))
            sector = None if terminated or truncated else env.last_objects[2]
            if progression is not None:
                progression.record(len(latencies), action, previous_obs, state, sector)
            if terminated or truncated:
                break
        metrics = {key: info[key] for key in ("kills", "keys_picked", "keys_total", "doors_opened", "doors_total",
                                             "locked_opened", "locked_total", "damage_taken", "stuck_steps",
                                             "dead", "cells_visited")}
        metrics.update(tier=tier, wad=path.name, map=map_name, seed=seed, exit=info["exit_reached"],
                       steps=len(latencies), base_reward=round(base_reward, 4), avg_latency_ms=float(np.mean(latencies)))
        metrics["score"] = round(sum(metrics[key] * weight for key, weight in SCORE.items()), 4)
        if progression is not None:
            counts = Counter(event["event"] for event in progression.events)
            metrics.update(events=progression.events, milestones={name: counts[name] for name in MILESTONES},
                           branch_junction_sector=progression.branch_junction,
                           spawn_sector=progression.spawn_sector, combat=progression.combat,
                           spawn_decisions=progression.spawn_decisions,
                           wall_follow_decisions=progression.wall_follow_decisions)
        geometry = env.env.mapinfo.lines
        walls = np.empty((0, 2, 2)) if geometry is None else np.stack((geometry.p1, geometry.p2), axis=1)
        return metrics, np.asarray(positions), walls
    finally:
        env.close()


def summarize(episodes):
    def ratio(key, total):
        denominator = sum(row[total] for row in episodes)
        return round(sum(row[key] for row in episodes) / denominator, 4) if denominator else None

    steps = sum(row["steps"] for row in episodes)
    summary = {
        "episodes": len(episodes),
        "avg_score": round(float(np.mean([row["score"] for row in episodes])), 4),
        "exit_rate": round(float(np.mean([row["exit"] for row in episodes])), 4),
        "door_rate": ratio("doors_opened", "doors_total"),
        "locked_door_rate": ratio("locked_opened", "locked_total"),
        "keycard_rate": ratio("keys_picked", "keys_total"),
        "stuck_step_pct": round(100 * sum(row["stuck_steps"] for row in episodes) / steps, 3),
        "avg_latency_ms": round(sum(row["avg_latency_ms"] * row["steps"] for row in episodes) / steps, 4),
    }
    if all("milestones" in row for row in episodes):
        summary["episodes_by_milestone"] = {
            name: sum(row["milestones"][name] > 0 for row in episodes) for name in MILESTONES
        }
    return summary


def trajectory(path, episode, positions, walls):
    width, height = 900, 680
    image = Image.new("RGB", (width, height), "#101722")
    draw = ImageDraw.Draw(image)
    label_font = ImageFont.load_default(size=18)
    body_font = ImageFont.load_default(size=14)
    draw.text((26, 19), f"{episode['wad']} / {episode['map']}  |  {episode['tier']}", "#edf5ff", font=label_font)
    draw.text((26, 46), f"Score {episode['score']:+.2f}    {episode['steps']} decisions    "
              f"{episode['cells_visited']} cells visited    exit {'yes' if episode['exit'] else 'no'}",
              "#a4b3c3", font=body_font)
    points = np.concatenate((positions, walls.reshape(-1, 2)), axis=0)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    span = np.maximum(maximum - minimum, 1)
    scale = min((width - 70) / span[0], (height - 150) / span[1])
    center = (minimum + maximum) / 2

    def pixel(point):
        x, y = (point - center) * scale
        return width / 2 + x, (height + 40) / 2 - y

    for a, b in walls:
        draw.line((pixel(a), pixel(b)), fill="#45576e", width=1)
    for index, (a, b) in enumerate(zip(positions[:-1], positions[1:])):
        fraction = index / max(len(positions) - 2, 1)
        draw.line((pixel(a), pixel(b)), fill=(int(36 + 215 * fraction), int(220 - 90 * fraction), 129), width=3)
    for point, color in ((positions[0], "#57eca4"), (positions[-1], "#ff9d72")):
        x, y = pixel(point)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="#f2f6ff", width=1)
    colors = {"door_approach": "#66ccff", "use_attempt": "#bfafff", "key_pickup": "#ffe06b",
              "branch_return": "#fafafa", "lock_approach": "#ff7ec5", "unlock": "#48e8df"}
    markers = {}
    for event in episode.get("events", ()):
        kind = event["event"]
        if kind not in colors or (kind == "use_attempt" and event["door_sector"] is None):
            continue
        key = kind, event.get("door_sector")
        previous = markers.get(key)
        if previous is None or (kind == "use_attempt" and event["in_use_ray"] and not previous["in_use_ray"]):
            markers[key] = event
    for event in markers.values():
        x, y = pixel((event["x"], event["y"]))
        color = colors[event["event"]]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="#101722")
    labels = {"door_approach": "Door approach", "use_attempt": "USE near door", "key_pickup": "Key pickup",
              "branch_return": "Branch return", "lock_approach": "Lock approach", "unlock": "Unlock"}
    legend = [("Start", "#57eca4"), ("End", "#ff9d72")]
    present = {event["event"] for event in markers.values()}
    legend.extend((labels[kind], color) for kind, color in colors.items() if kind in present)
    x = 26
    for label, color in legend:
        draw.text((x, height - 30), label, color, font=body_font)
        x += draw.textlength(label, font=body_font) + 22
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=V1_WEIGHTS_PATH)
    parser.add_argument("--maps", type=Path, default=ROOT / "maps" / "eval")
    parser.add_argument("--obsidian-eval", type=Path, help="score the 12 OBSIDIAN holdouts instead of the procedural tiers")
    parser.add_argument("--legacy-freedoom", action="store_true", help="score only MAP01 and MAP02 for older comparisons")
    parser.add_argument("--report", type=Path, default=ROOT / "eval" / "eval_report.json")
    parser.add_argument("--plots", type=Path, default=ROOT / "eval" / "plots")
    parser.add_argument("--ticks", type=int, default=2100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fleet-seeds", action="store_true", help="use the fleet's fixed per-map seeds")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.ticks < 4 or args.seed < 0:
        parser.error("Provide a checkpoint, at least four ticks, and a nonnegative seed")
    jobs = []
    if args.obsidian_eval:
        jobs.extend((path, name, "obsidian") for path, name in obsidian_maps(args.obsidian_eval, "eval"))
    else:
        for tier in TIERS:
            maps = sorted((args.maps / tier).glob("*.wad"))
            if len(maps) != 3:
                parser.error(f"Expected three maps in {args.maps / tier}")
            jobs.extend((path, "MAP01", tier) for path in maps)
    freedoom = freedoom_wad()
    if freedoom.is_file():
        names = FREEDOOM_EVAL[:2] if args.legacy_freedoom else FREEDOOM_EVAL
        jobs.extend((freedoom, name, "freedoom2") for name in names)
    from doom_env import action_space, observation_space

    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    agent = DoomAgent(observation_space(), action_space(), "cpu")
    agent.load_checkpoint(args.checkpoint)
    episodes, skipped = [], []
    for index, (wad, map_name, tier) in enumerate(jobs):
        seed = (int.from_bytes(hashlib.sha256(f"{tier}/{wad.name}/{map_name}".encode()).digest()[:4], "big")
                if args.fleet_seeds else args.seed + index)
        try:
            result = run_episode(agent, wad, map_name, tier, seed, args.ticks)
        except (RuntimeError, KeyError) as error:
            if tier != "freedoom2":
                raise
            skipped.append({"wad": wad.name, "map": map_name, "error": str(error)})
            continue
        episodes.append(result)
        row = result[0]
        print(f"{tier:17} {wad.name}:{map_name} | score {row['score']:+.2f} | "
              f"exit {row['exit']} | steps {row['steps']}", flush=True)
    args.plots.mkdir(parents=True, exist_ok=True)
    for row, points, walls in episodes:
        folder = args.plots / row["tier"]
        folder.mkdir(exist_ok=True)
        name = f"{Path(row['wad']).stem}_{row['map']}.png"
        trajectory(folder / name, row, points, walls)
        row["tier_trajectory_plot"] = (folder / name).relative_to(args.plots).as_posix()
    for index, (row, points, walls) in enumerate(sorted(episodes, key=lambda result: result[0]["score"])[:4], 1):
        name = f"worst_{index:02d}.png"
        trajectory(args.plots / name, row, points, walls)
        row["trajectory_plot"] = name
    rows = [result[0] for result in episodes]
    report = {"checkpoint": args.checkpoint.name, "policy": "v1" if agent.spatial else "v0",
              "checkpoint_sha256": checkpoint_hash,
              "episode_ticks": args.ticks, "score_weights": SCORE, "episodes": rows, "skipped": skipped,
              "seed_mode": "fleet" if args.fleet_seeds else "sequential", "metrics": summarize(rows),
              "tiers": {tier: summarize([row for row in rows if row["tier"] == tier])
                        for tier in (*TIERS, "obsidian", "freedoom2") if any(row["tier"] == tier for row in rows)}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.report} and {len(rows)} per-tier trajectories in {args.plots}")


if __name__ == "__main__":
    main()
