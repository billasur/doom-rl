"""Train or resume a bounded v1 burst, optionally distilling frozen teachers."""

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_env] = (os.environ.get("DOOM_CPU_LEARNER_THREADS", "1")
                               if __name__ == "__main__" and _thread_env != "OPENBLAS_NUM_THREADS" else "1")

import gymnasium as gym
import numpy as np
import torch
import yaml
from doom_env import observation_space

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from models.v1_spatial_net import SpatialActorCritic, SpatialState, checkpoint_features, load_spatial_weights, mask_use
from maps.gen_wads import textmap, write_udmf_pack
from maps.micro_drills import lock_drill
from maps.pool import FREEDOOM_TRAIN, V2_FREEDOOM_TRAIN, freedoom_wad, obsidian_corridors, obsidian_maps
from train.batched_rollout import BatchedRolloutPolicy
from train.fusion import guided_kl_weight
from train.checkpoint_eval import SPECIALIST_TIERS, CheckpointEval
from train.train_v1 import TIERS, CurriculumEnv, collect, isolated_vizdoom_worker, ppo_update, precision, tensor

SAVE_RESERVE = 45


def atomic_save(path, payload, *, json_file=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        if json_file:
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        else:
            torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def cuda_amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def restore_adam(optimizer, snapshot, learning_rate, *, added_features=False):
    if not snapshot:
        return False
    try:
        saved_groups = snapshot["param_groups"]
        if len(saved_groups) != len(optimizer.param_groups):
            return False
        groups = []
        live_state = optimizer.state_dict()
        for saved, current, live in zip(saved_groups, optimizer.param_groups, live_state["param_groups"]):
            if (len(saved["params"]) > len(current["params"])
                    or (len(saved["params"]) != len(current["params"]) and not added_features)):
                return False
            for key, param in zip(saved["params"], current["params"]):
                state = snapshot["state"].get(key, {})
                if state and (state["step"].numel() != 1 or any(
                    state[name].shape != param.shape for name in ("exp_avg", "exp_avg_sq")
                )):
                    return False
            groups.append({**saved, "params": live["params"]})
        mapped = {new: snapshot["state"][old] for saved, group in zip(saved_groups, groups)
                  for old, new in zip(saved["params"], group["params"]) if old in snapshot["state"]}
        optimizer.load_state_dict({"param_groups": groups, "state": mapped})
    except (KeyError, ValueError, TypeError, AttributeError):
        return False
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return True


def load_initial(path, model, optimizer, moments, learning_rate, *, select_best=False, scaler=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model" not in payload:
        if select_best:
            raise ValueError("--resume-best expects a full state.pt checkpoint")
        load_spatial_weights(model, payload, allow_new=True)
        return {}, dict(mode="weights_only", optimizer_restored=False, normalizer_count=0)
    selected = payload.get("best_model") if select_best else None
    weights = selected if selected is not None else payload["model"]
    added_features = set(model.state_dict()) > set(weights)
    load_spatial_weights(model, weights, allow_new=True)
    saved_moments = payload["obs_normalizer"]
    if saved_moments["count"] < 0:
        raise ValueError("Checkpoint observation count must be nonnegative")
    for key in ("mean", "m2"):
        old, current = saved_moments[key], moments[key]
        if old.shape == current.shape or (added_features and old.ndim == 1 and old.numel() < current.numel()):
            current[:old.numel()].copy_(old)
        else:
            raise ValueError("Checkpoint observation statistics do not match spatial telemetry")
    moments["count"] = saved_moments["count"]
    optimizer_state = payload.get("best_optimizer") if selected is not None else payload.get("optimizer")
    if selected is not None and optimizer_state is None and all(
        torch.equal(tensor, payload["model"][key]) for key, tensor in selected.items()
    ):
        optimizer_state = payload.get("optimizer")
    restored = restore_adam(optimizer, optimizer_state, learning_rate, added_features=added_features)
    if restored and selected is None and scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if not select_best and "rng" in payload:
        torch.set_rng_state(payload["rng"])
    return payload, dict(mode="best_model_with_state" if selected is not None else "full_state",
                         optimizer_restored=restored, expanded_policy=added_features,
                         normalizer_count=moments["count"])


def update_moments(moments, telemetry):
    """Track packed-observation moments while retaining v1's fixed input scaling."""
    positions = telemetry.flatten(0, 1).double()
    count, old_count = len(positions), moments["count"]
    mean = positions.mean(dim=0).cpu()
    m2 = ((positions - mean.to(positions.device)).square().sum(dim=0)).cpu()
    delta = mean - moments["mean"]
    total = old_count + count
    moments["mean"] += delta * count / total
    moments["m2"] += m2 + delta.square() * (old_count * count / total)
    moments["count"] = total


def reset_group(env, model, device, seed=None):
    obs, info = env.reset(seed=seed)
    starts = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    state = model.initial_state(tensor(obs, device))
    stats = {"returns": np.zeros(env.num_envs), "recent": deque(maxlen=32),
             "cells": deque(maxlen=32), "exits": deque(maxlen=32), "tiers": np.zeros(len(TIERS), dtype=np.int64)}
    return [obs, info, state, starts, stats]


def rollout(env, model, group, config, device, stream=None):
    obs, info, state, starts, stats = group
    with (torch.cuda.stream(stream) if stream is not None else nullcontext()), precision(device, config):
        batch, obs, info, state, starts = collect(env, model, obs, info, state, starts, config, device, stats)
    return batch, [obs, info, state, starts, stats]


def combine(parts):
    batch = {key: torch.cat([part[key] for part in parts], dim=1) for key in parts[0] if key != "states"}
    batch["states"] = SpatialState(*(torch.cat(items, dim=1) for items in zip(*(part["states"] for part in parts))))
    advantage = batch["returns"] - batch["values"]
    batch["advantages"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    return batch


@torch.no_grad()
def distill(teachers, states, priors, batch, device, config):
    predictions, values = [], []
    guided = "teacher_domain_weights" in config
    temperatures = config.get("teacher_temperatures", [1.0] * len(teachers))
    for index, teacher in enumerate(teachers):
        state = states[index]
        probs, critic_values = [], []
        for tick, (observation, privileged, starts) in enumerate(zip(batch["obs"], batch["privileged"], batch["starts"])):
            state = teacher.initial_state(observation) if state is None else state.reset_where(starts)
            with precision(device, config):
                logits, value, _, state = teacher(observation, state, None if guided else privileged)
            probs.append(mask_use(logits.float() / temperatures[index], batch["prev_used"][tick]).softmax(-1) if teacher.categorical
                         else logits.float().sigmoid())
            if not guided:
                critic_values.append(value.float())
        states[index] = state
        predictions.append(torch.stack(probs))
        if not guided:
            values.append(torch.stack(critic_values))
    predictions = torch.stack(predictions)
    if guided:
        domains = batch["teacher_domain"]
        if config.get("route_any_held_key"):
            domains = torch.where(batch["teacher_key_held"], 4, domains)
        weights = priors[:, domains]
        postkey = torch.as_tensor(config["teacher_postkey_evidence"], dtype=torch.bool, device=device)
        qualified = ~batch["teacher_key_held"] | postkey[:, domains]
        repeated_mistake = (predictions.argmax(-1) == batch["actions"]) & batch["teacher_bad_transition"]
        eligible_before_action = (weights > 0) & qualified
        weights = weights.masked_fill(~qualified | repeated_mistake, 0)
        total = weights.sum(0)
        batch["teacher_valid"] = total > 0
        mixture = weights / total.clamp_min(torch.finfo(weights.dtype).tiny)
    else:
        mixture = (torch.stack(values).clamp(-20, 20) + priors[:, batch["tier"]].log()).softmax(dim=0)
    batch["teacher_probs"] = (predictions * mixture[..., None]).sum(dim=0)
    if guided:
        # Count complete rollout decisions and distinguish abstention from a useful zero KL.
        decisions = torch.stack((torch.ones_like(total, dtype=torch.bool), batch["teacher_valid"],
                                 (eligible_before_action & repeated_mistake).any(0))).long().flatten(1)
        counts = torch.zeros((3, 7), dtype=torch.long, device=device)
        return counts.scatter_add_(1, domains.reshape(1, -1).expand_as(decisions), decisions)


def train(args, config, started):
    deadline = started + args.wall_budget_sec - SAVE_RESERVE
    device = torch.device(args.device)
    learner_threads = int(os.environ.get("DOOM_CPU_LEARNER_THREADS", "1")) if device.type == "cpu" else 1
    rollout_threads = int(os.environ.get("DOOM_CPU_ROLLOUT_THREADS", "1")) if device.type == "cpu" else 1
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    amp_dtype = cuda_amp_dtype(device)
    if amp_dtype is not None:
        config["amp_dtype"] = amp_dtype
    if device.type == "cuda" and "cuda_memory_fraction" in config:
        torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"])
    scaler = torch.amp.GradScaler("cuda") if config.get("amp_dtype") == torch.float16 else None
    maps = [tuple(sorted((args.maps / tier).resolve().glob("*.wad"))) for tier in TIERS]
    stages = config["stages"]
    features = {name: config.get(name, False) for name in ("pose_aux", "action_history", "depth", "visits", "categorical")}
    generalist = config.get("reward_mode") == "v2_generalist"
    episode_mix = config.get("fusion_episode_mix")
    drill_prob = (episode_mix or {}).get("keycard", config.get("keycard_drill_prob", 0.0))
    if any(not paths for paths in maps):
        raise ValueError("All four training tiers must contain WADs")
    monster_density = config.get("monster_density", 1.0)
    if monster_density != 1.0:
        for tier, paths in enumerate(maps):
            pack = args.out_dir / f"procedural_{tier + 1}.wad"
            seeds = range(55000 + tier * 1000, 55000 + tier * 1000 + len(paths))
            write_udmf_pack(pack, (textmap(tier, seed, monster_density=monster_density) for seed in seeds))
            maps[tier] = tuple((pack, f"MAP{i:02}") for i in range(1, len(paths) + 1))
    tier4_groups, tier4_weights, tier4_names = [maps[-1]], [0.50], ["procedural"]
    explore_maps = ()
    if args.obsidian_train and config.get("obsidian_role") == "explore" and not generalist:
        explore_maps = obsidian_corridors(args.obsidian_train)
    elif args.obsidian_train:
        tier4_groups.append(obsidian_maps(args.obsidian_train, "train", compact_count=config.get("obsidian_compact_count")))
        tier4_weights.append(0.35)
        tier4_names.append("obsidian")
    if config.get("mix_freedoom", True):
        freedoom = freedoom_wad()
        if not freedoom.is_file():
            raise FileNotFoundError(freedoom)
        tier4_groups.append(tuple((freedoom, name) for name in (V2_FREEDOOM_TRAIN if generalist else FREEDOOM_TRAIN)))
        tier4_weights.append(0.15)
        tier4_names.append("freedoom")
    if generalist:
        mix = {"procedural": 0.25, "obsidian": 0.40, "freedoom": 0.35}
        tier4_weights = [mix[name] for name in tier4_names]
    tier4_sources = (tier4_groups, [weight / sum(tier4_weights) for weight in tier4_weights])
    warmup_maps = ()
    if any(stage.get("tier4_warmup") for stage in stages):
        pack = args.out_dir / "tier4_warmup.wad"
        write_udmf_pack(pack, (textmap(3, seed, combat_warmup=True) for seed in range(14000, 14016)))
        warmup_maps = tuple((pack, f"MAP{index:02}") for index in range(1, 17))
    combat_maps = ()
    if any(stage.get("tier4_combat_drill") for stage in stages):
        pack = args.out_dir / "tier4_ambush.wad"
        write_udmf_pack(pack, (textmap(3, seed, combat_drill=True) for seed in range(16000, 16024)))
        combat_maps = tuple((pack, f"MAP{index:02}") for index in range(1, 25))
    reverse_maps = {}
    for stage, offset in (("lock", 21000), ("branch", 22000), ("micro_lock", 35000)):
        if any(item.get("reverse_stage") == stage for item in stages) or (drill_prob and stage in ("micro_lock", "branch")):
            pack = args.out_dir / f"tier3_{stage}.wad"
            maps_for_stage = (lock_drill(seed) if stage == "micro_lock" else textmap(2, seed, reverse_stage=stage)
                              for seed in range(offset, offset + 24))
            write_udmf_pack(pack, maps_for_stage)
            reverse_maps[stage] = tuple((pack, f"MAP{index:02}") for index in range(1, 25))
    if config.get("fresh_keycard_layouts"):
        pack = args.out_dir / "tier3_full.wad"
        write_udmf_pack(pack, (textmap(2, seed) for seed in range(45000, 45048)))
        maps[2] = tuple((pack, f"MAP{index:02}") for index in range(1, 49))
    worker = partial(CurriculumEnv, maps, stages[0]["weights"], config["episode_ticks"],
                     1 if generalist else config.get("episodes_per_wad", 1), config.get("reward"),
                     warmup_maps=warmup_maps, warmup=stages[0].get("tier4_warmup", False),
                     tier4_sources=tier4_sources, tier4_reward=config.get("tier4_reward"),
                     reverse_maps=reverse_maps, reverse_stage=stages[0].get("reverse_stage"),
                     combat_maps=combat_maps, combat_drill=stages[0].get("tier4_combat_drill", False),
                     explore_maps=explore_maps, reward_mode=config.get("reward_mode"),
                     movement_penalty=config.get("movement_penalty", False),
                     generalist_sources=tier4_sources if generalist else None, keycard_drill_prob=drill_prob,
                     branch_drills_only=config.get("branch_drills_only", False),
                     fusion_episode_mix=episode_mix, source_names=tier4_names,
                     freedoom_random_start=config.get("freedoom_random_start", False),
                     label_teacher_domains="teacher_domain_weights" in config, **features)
    features["reset_visit_on_key"] = config.get("reset_visit_on_key", False)
    envs = []
    cancelled = False

    def stop(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        count = config["workers"]
        collector_groups = config.get("collector_groups", 2)
        start_method = config.get("env_start_method", "spawn")
        if start_method == "forkserver":
            mp.set_forkserver_preload(["train.train_v1"])
        vector_type = gym.vector.AsyncVectorEnv
        if config.get("batch_cpu_info") and device.type == "cpu":
            from tools.arena_cpu_vector import BatchedInfoVectorEnv
            vector_type = BatchedInfoVectorEnv
        for i in range(collector_groups):
            size = count // collector_groups + (i >= collector_groups - count % collector_groups)
            envs.append(vector_type(
                [worker] * size, context=start_method, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
                worker=isolated_vizdoom_worker,
            ))
        spec = observation_space(), envs[0].single_action_space
        model = SpatialActorCritic(*spec, **features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-5)
        steps, best_score, best_weights, best_optimizer, best_step = 0, None, None, None, None
        evaluator = CheckpointEval(spec, args.out_dir / "checkpoint_eval", config.get("checkpoint_eval_ticks", 840),
                                   features=features, tier_weights=config.get("checkpoint_tier_weights"),
                                   role=config.get("checkpoint_role", "mixed"))
        groups = [reset_group(env, model, device, config["seed"] + i * count) for i, env in enumerate(envs)]
        width = groups[0][0].shape[-1]
        moments = {"count": 0, "mean": torch.zeros(width, dtype=torch.float64),
                   "m2": torch.zeros(width, dtype=torch.float64)}
        initialization = dict(mode="fresh", optimizer_restored=False, normalizer_count=0)
        if args.init_from:
            checkpoint, initialization = load_initial(args.init_from, model, optimizer, moments,
                config["learning_rate"], select_best=args.resume_best, scaler=scaler)
            if checkpoint:
                steps = checkpoint["best_eval_step"] if args.resume_best and checkpoint.get("best_model") else checkpoint["global_step"]
                if (not args.resume_best and checkpoint.get("best_eval_protocol") == evaluator.protocol
                        and checkpoint.get("best_model") and set(checkpoint["best_model"]) == set(model.state_dict())):
                    best_score, best_weights = checkpoint["best_eval_score"], checkpoint["best_model"]
                    best_optimizer = checkpoint.get("best_optimizer")
                    best_step = checkpoint["best_eval_step"]
        if config.get("fresh_actor"):
            model.actor.reset_parameters()
            for parameter in model.actor.parameters():
                optimizer.state.pop(parameter, None)
            initialization["fresh_actor"] = True
            best_score, best_weights, best_optimizer, best_step = None, None, None, None
        print(f"{args.run_id} | {device} | worker groups {[env.num_envs for env in envs]} | "
              f"rollout threads {rollout_threads} | PPO threads {learner_threads} | "
              f"{initialization['mode']} at step {steps} | optimizer restored "
              f"{initialization['optimizer_restored']} | observation count {initialization['normalizer_count']}", flush=True)
        teachers = []
        for filename in args.teachers:
            weights = torch.load(filename, map_location="cpu", weights_only=True)
            weights = weights.get("model", weights)
            # Submission exports may carry their sampled temperature as metadata.
            weights = {k: v for k, v in weights.items() if k != "_policy_temperature"}
            teacher = SpatialActorCritic(*spec, **checkpoint_features(weights)).to(device)
            if teacher.categorical != model.categorical:
                raise ValueError("Teacher and student action spaces must match")
            teacher.load_state_dict(weights)
            teachers.append(teacher.eval().requires_grad_(False))
        teacher_states = [None] * len(teachers)
        if "teacher_domain_weights" in config:
            priors = torch.as_tensor(config["teacher_domain_weights"], dtype=torch.float32, device=device)
            temps = torch.as_tensor(config.get("teacher_temperatures", []), dtype=torch.float32)
            evidence = torch.as_tensor(config.get("teacher_postkey_evidence", []))
            if (not teachers or not model.categorical or priors.shape != (len(teachers), 7)
                    or not torch.isfinite(priors).all() or (priors < 0).any()
                    or temps.shape != (len(teachers),) or not torch.isfinite(temps).all() or (temps <= 0).any()
                    or evidence.shape != priors.shape or evidence.dtype != torch.bool):
                raise ValueError("Provide seven domain weights and unlock flags, and a positive temperature per categorical teacher")
        else:
            priors = torch.as_tensor(config.get("teacher_tier_weights", [[1] * 4] * len(teachers)),
                                     dtype=torch.float32, device=device)
            if teachers and (priors.shape != (len(teachers), 4) or not torch.isfinite(priors).all()
                             or (priors <= 0).any()):
                raise ValueError("teacher_tier_weights must supply four positive weights per teacher")

        teacher_counts = (torch.zeros((3, 7), dtype=torch.long, device=device)
                          if "teacher_domain_weights" in config else None)
        eval_count = 0
        evaluated_step = None
        last_eval = None
        timings = dict(rollout=0.0, batch_and_distill=0.0, ppo=0.0, checkpoint_eval=0.0)

        def check_weights():
            nonlocal best_score, best_weights, best_optimizer, best_step, eval_count, evaluated_step, last_eval
            if steps == evaluated_step:
                return
            checking = time.monotonic()
            weights = cpu_tree(model.state_dict())
            score, scores = evaluator.evaluate(weights)
            timings["checkpoint_eval"] += time.monotonic() - checking
            evaluated_step = steps
            eval_count += 1
            last_eval = dict(step=steps, score=score, tiers=scores)
            if best_score is None or score > best_score:
                best_score, best_weights, best_step = score, weights, steps
                best_optimizer = cpu_tree(optimizer.state_dict())
            print(f"checkpoint eval | step {steps} | score {score:+.4f} | best {best_score:+.4f} "
                  f"at {best_step} | {scores}", flush=True)

        def save(snapshot=None):
            current = cpu_tree(model.state_dict())
            payload = {
                "model": current, "optimizer": cpu_tree(optimizer.state_dict()), "obs_normalizer": moments,
                "global_step": steps, "run_initial_step": run_origin,
                "best_eval_score": best_score, "best_model": best_weights,
                "best_optimizer": best_optimizer, "best_eval_step": best_step, "best_eval_protocol": evaluator.protocol,
                "rng": torch.get_rng_state(), "scaler": scaler.state_dict() if scaler is not None else None,
            }
            atomic_save(args.out_dir / "state.pt", payload)
            atomic_save(args.out_dir / "best_weights.pt", best_weights)
            if snapshot is not None:
                folder = (args.progress_dir or args.out_dir) / "snapshots" / f"{snapshot:09d}"
                folder.mkdir(parents=True, exist_ok=True)
                atomic_save(folder / "state.pt", payload)
                atomic_save(folder / "weights.pt", current)
                atomic_save(folder / "snapshot.json", {"step": steps, "threshold": snapshot}, json_file=True)

        initial_steps = steps
        run_origin = checkpoint.get("run_initial_step", steps) if args.resume_output and args.init_from and checkpoint else steps
        stage = checks_without_progress = 0
        early_stopped = False
        peak_reward = peak_cells = -float("inf")
        last_check = next_save = time.monotonic()
        progress_interval = config.get("eval_interval_seconds", 8)
        checkpoint_eval_interval = config.get("checkpoint_eval_every_seconds", 60)
        min_stop_steps = config.get("min_steps_before_early_stop", 350_000)
        reward = cells = None
        kl, kl_weight, updates = None, 0.0, 0
        guided_fraction = 0.0
        budget_start = time.monotonic()
        last_rollout_sec = 0.0
        target_steps = config.get("burst_steps")
        snapshot_interval = config.get("snapshot_every_steps", 0)
        next_snapshot = (steps // snapshot_interval + 1) * snapshot_interval if snapshot_interval else None
        initial_entropy = config["entropy_weight"]
        final_entropy = config.get("final_entropy_weight", initial_entropy)
        initial_lr = config["learning_rate"]
        final_lr = config.get("final_learning_rate", initial_lr)
        ppo_precision = {}
        check_weights()
        torch.set_num_threads(rollout_threads)
        next_eval = time.monotonic() + checkpoint_eval_interval
        batch_rollouts = config.get("batch_rollout_inference", False)
        parallel_cuda = device.type == "cuda" and not batch_rollouts and config.get("parallel_cuda_rollout", False)
        streams = [torch.cuda.Stream() for _ in envs] if parallel_cuda else [None] * len(envs)
        actor = BatchedRolloutPolicy(model, config, device, len(envs)) if batch_rollouts else nullcontext(model)
        with actor as rollout_model, ThreadPoolExecutor(max_workers=len(envs)) as pool:
            while not cancelled:
                now = time.monotonic()
                at_cutoff = now + last_rollout_sec >= deadline
                # Keep one fresh, teacher-free PPO batch when wall-clock jitter ends the ramp early.
                final_ppo = (at_cutoff and not target_steps and config.get("kl_decay_fraction")
                             and updates and kl_weight > 0 and now < deadline + SAVE_RESERVE - 12
                             and now + last_rollout_sec < deadline + SAVE_RESERVE - 6)
                if at_cutoff and not final_ppo:
                    break
                done_steps = steps - run_origin
                if target_steps is not None and done_steps >= target_steps:
                    break
                progress = (done_steps / target_steps if target_steps
                            else (now - budget_start) / max(deadline - budget_start, 1e-6))
                config["entropy_weight"] = initial_entropy + min(progress, 1.0) * (final_entropy - initial_entropy)
                if final_lr != initial_lr:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = initial_lr + min(progress, 1.0) * (final_lr - initial_lr)
                next_stage = max(i for i, item in enumerate(stages) if item["start"] <= progress)
                if next_stage != stage:
                    for env in envs:
                        item = stages[next_stage]
                        env.call("set_stage", item["weights"], item.get("tier4_warmup", False),
                                 item.get("reverse_stage"), item.get("tier4_combat_drill", False))
                    groups = [reset_group(env, model, device) for env in envs]
                    teacher_states = [None] * len(teachers)
                    peak_reward = peak_cells = -float("inf")
                    checks_without_progress = 0
                    stage = next_stage
                    print(f"curriculum | stage {stage + 1}/{len(stages)} | step {steps} | {item}", flush=True)
                collection_started = time.monotonic()
                if parallel_cuda:
                    for stream in streams:
                        stream.wait_stream(torch.cuda.current_stream())
                jobs = [pool.submit(rollout, env, rollout_model, group, config, device, stream)
                        for env, group, stream in zip(envs, groups, streams)]
                parts, groups = zip(*(job.result() for job in jobs))
                if parallel_cuda:
                    # Rollout buffers and recurrent states move between the collector and learner streams.
                    torch.cuda.synchronize()
                collected = time.monotonic()
                timings["rollout"] += collected - collection_started
                batch = combine(parts)
                update_moments(moments, batch["obs"])
                if teachers:
                    # With no explicit step target, forecast the remaining steps from observed throughput.
                    per_update = count * config["rollout_steps"]
                    fps = (done_steps + per_update) / max(time.monotonic() - budget_start, 1e-6)
                    planned = target_steps or max(done_steps + per_update, fps * (deadline - budget_start))
                    step_fraction = min(1.0, (done_steps + per_update) / planned)
                    if "teacher_domain_weights" in config:
                        fraction = progress
                        if config.get("kl_decay_fraction"):
                            if not target_steps and last_rollout_sec:
                                # Count how many complete PPO batches still fit, including this one.
                                batches_left = max(1, int((deadline - now) / last_rollout_sec))
                                step_fraction = (done_steps + per_update) / (done_steps + batches_left * per_update)
                            guided_fraction = 1.0 if final_ppo else max(guided_fraction, step_fraction)
                            fraction = guided_fraction
                        kl_weight = guided_kl_weight(config.get("kl_start", 0.35), fraction,
                                                     config.get("kl_decay_fraction", 1.0))
                    else:
                        kl_weight = config.get("kl_start", 0.15) * min(1.0, (1 - step_fraction) / 0.3)
                    if not final_ppo:
                        counts = distill(teachers, teacher_states, priors, batch, device, config)
                        if teacher_counts is not None:
                            teacher_counts += counts
                updating = time.monotonic()
                timings["batch_and_distill"] += updating - collected
                if learner_threads > 1:
                    torch.set_num_threads(learner_threads)
                try:
                    result = ppo_update(model, optimizer, batch, config, device, scaler=scaler, kl_weight=kl_weight,
                                        profile=ppo_precision)
                finally:
                    if learner_threads > 1:
                        torch.set_num_threads(rollout_threads)
                timings["ppo"] += time.monotonic() - updating
                if isinstance(result, tuple):
                    kl = result[1]
                steps += count * config["rollout_steps"]
                updates += 1
                last_rollout_sec = time.monotonic() - now
                if (snapshot_interval or config.get("live_progress")) and (
                    updates == 1 or time.monotonic() - last_check >= progress_interval
                ):
                    fps = (steps - initial_steps) / (time.monotonic() - budget_start)
                    if device.type == "cpu":
                        print(f"{args.run_id} | step {steps} | {fps:.1f} steps/sec | "
                              f"rollout {collected - collection_started:.2f}s | PPO {time.monotonic() - updating:.2f}s", flush=True)
                    else:
                        print(f"{args.run_id} | step {steps} | {fps:.1f} steps/sec", flush=True)
                    live = dict(step=steps, steps_per_sec=round(fps, 1), updated_at=time.time(),
                                device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                                rollout_sec=round(collected - collection_started, 3),
                                ppo_sec=round(time.monotonic() - updating, 3))
                    if teacher_counts is not None:
                        live.update(kl_weight=kl_weight, teacher_count=len(teachers))
                    if device.type == "cuda":
                        live.update(gpu_peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 1024**2),
                                    gpu_reserved_mib=round(torch.cuda.memory_reserved() / 1024**2))
                    if batch_rollouts:
                        live["rollout_requests_per_forward"] = round(actor.calls / max(actor.batches, 1), 2)
                    atomic_save(args.out_dir / "live.json", live, json_file=True)
                    if args.progress_dir is not None:
                        atomic_save(args.progress_dir / "live.json", live, json_file=True)
                recent_rewards = [value for group in groups for value in group[4]["recent"]]
                recent_cells = [value for group in groups for value in group[4]["cells"]]
                if recent_rewards:
                    reward, cells = float(np.mean(recent_rewards)), float(np.mean(recent_cells))
                    if time.monotonic() - last_check >= progress_interval:
                        can_stop = steps - initial_steps >= min_stop_steps
                        if can_stop:
                            improved = reward > peak_reward + 1e-3 or cells > peak_cells + 1e-3
                            peak_reward, peak_cells = max(peak_reward, reward), max(peak_cells, cells)
                            checks_without_progress = 0 if improved else checks_without_progress + 1
                        last_check = time.monotonic()
                        print(f"step {steps} | return {reward:.3f} | cells {cells:.1f} | KL {kl} @ {kl_weight:.3g}", flush=True)
                        if can_stop and 0 < args.early_stop_patience <= checks_without_progress:
                            early_stopped = True
                            break
                improved_eval = False
                if time.monotonic() >= next_eval:
                    previous_best = best_step
                    check_weights()
                    improved_eval = best_step != previous_best
                    next_eval = time.monotonic() + checkpoint_eval_interval
                if next_snapshot is not None and steps >= next_snapshot:
                    save(next_snapshot)
                    next_snapshot += snapshot_interval
                    next_save = time.monotonic() + min(120, config["checkpoint_every_seconds"])
                elif improved_eval or time.monotonic() >= next_save:
                    save()
                    next_save = time.monotonic() + min(120, config["checkpoint_every_seconds"])
        check_weights()
        save()
        atomic_save(args.out_dir / "final_weights.pt", cpu_tree(model.state_dict()))
        elapsed = time.monotonic() - started
        if not updates and not (args.resume_output and target_steps and steps - run_origin >= target_steps):
            raise RuntimeError("Budget expired before a PPO update")
        source_episodes = sum(group[1]["tier4_source_episodes"].sum(axis=0) for group in groups).tolist()
        drills = sum(group[1]["drill_episodes"].sum(axis=0) for group in groups).tolist()
        summary = {"run_id": args.run_id, "steps": steps - run_origin, "global_step": steps,
                   "resumed_from_step": initial_steps, "initialization": initialization,
                   "fps": (steps - initial_steps) / elapsed,
                   "wall_sec": elapsed, "early_stopped": early_stopped, "final_reward": reward,
                   "explored_cells": cells, "teachers": len(teachers), "last_kl": kl,
                   "last_kl_weight": kl_weight, "best_eval_score": best_score,
                   "best_eval_step": best_step, "best_eval_protocol": evaluator.protocol,
                   "checkpoint_evaluations": eval_count, "last_checkpoint_eval": last_eval,
                   "timing_sec": timings, "updates": updates,
                   "rollout_steps_per_env": config["rollout_steps"],
                   "sequence_steps": config["sequence_steps"],
                   "minibatches_per_epoch": int(np.ceil(count * config["rollout_steps"] /
                                                        (config["sequence_steps"] * config["minibatch_sequences"]))),
                   "ppo_epochs": config["epochs"],
                   "amp_dtype": str(config["amp_dtype"]) if "amp_dtype" in config else None,
                   "ppo_precision": ppo_precision,
                   "min_steps_before_early_stop": min_stop_steps,
                   "wad_loads": sum(int(group[1]["wad_loads"].sum()) for group in groups),
                   "episodes_started": sum(int(group[1]["episodes_started"].sum()) for group in groups),
                   "tier4_warmup_episodes": sum(int(group[1]["warmup_episodes"].sum()) for group in groups),
                   "tier4_source_counts": dict(zip(tier4_names, map(len, tier4_groups))),
                   "tier4_source_episodes": dict(zip(tier4_names, source_episodes)),
                   "drill_episodes": dict(zip(("key_near_lock", "near_branch_key", "tier4_ambush", "obsidian_corridors"), drills)),
                   "obsidian_corridor_maps": len(explore_maps), "policy_features": features,
                   "best_weights_path": str(args.out_dir / "best_weights.pt"),
                   "final_weights_path": str(args.out_dir / "final_weights.pt")}
        if episode_mix is not None:
            bucket_counts = [c for env in envs for c in env.call("fusion_counts")]
            summary["fusion_episode_counts"] = {name: sum(c[name] for c in bucket_counts) for name in episode_mix}
        if teacher_counts is not None:
            summary["teacher_domain_decisions"] = dict(zip(("seen", "imitated", "rejected_bad_agreement"),
                                                           teacher_counts.cpu().tolist()))
        atomic_save(args.out_dir / "train_summary.json", summary, json_file=True)
        print(json.dumps(summary), flush=True)
    finally:
        for env in envs:
            env.close(timeout=3)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v1.yaml")
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume-best", action="store_true",
                        help="select the best model from state.pt and begin fresh best-score tracking")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--resume-output", action="store_true", help="resume an interrupted run in its existing output directory")
    parser.add_argument("--progress-dir", type=Path, help="continue the snapshot and progress stream of a stopped arm")
    parser.add_argument("--wall-budget-sec", required=True, type=float)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--min-steps-before-early-stop", type=int)
    parser.add_argument("--teachers", type=lambda x: [Path(p.strip()) for p in x.split(",") if p.strip()], default=[])
    parser.add_argument("--maps", type=Path, default=ROOT / "maps" / "train")
    parser.add_argument("--obsidian-train", type=Path, help="folder with the OBSIDIAN training manifest and packs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.resume_best and not args.init_from:
        parser.error("--resume-best requires --init-from")
    if not SAVE_RESERVE < args.wall_budget_sec < float("inf") or args.early_stop_patience < 0:
        parser.error("Use a budget above the 45-second save reserve and nonnegative patience")
    if not args.resume_output and args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Use a fresh output directory and pass --init-from to resume")
    config = yaml.safe_load((ROOT / "configs" / "v1.yaml").read_text())
    raw_config = args.config.read_text()
    try:
        overrides = json.loads(raw_config)
    except json.JSONDecodeError:
        overrides = yaml.safe_load(raw_config)
    config.update(overrides or {})
    if args.min_steps_before_early_stop is not None:
        config["min_steps_before_early_stop"] = args.min_steps_before_early_stop
    stages = config["stages"]
    fractions = [stage["start"] for stage in stages]
    checkpoint_weights = config.get("checkpoint_tier_weights", [1] * len(TIERS))
    if (config["workers"] < 2 or not 2 <= config.get("collector_groups", 2) <= config["workers"]
            or config.get("env_start_method", "spawn") not in mp.get_all_start_methods()
            or (sys.platform == "darwin" and config.get("env_start_method", "spawn") != "spawn")
            or not 0 < config.get("cuda_memory_fraction", 1.0) <= 1.0
            or config["sequence_steps"] < 1 or config["rollout_steps"] < 1
            or not 0 <= config.get("keycard_drill_prob", 0) <= 1
            or not 0 < config.get("monster_density", 1.0) <= 1
            or (config.get("keycard_drill_prob", 0) and config.get("reward_mode") != "v2_generalist")
            or config["rollout_steps"] % config["sequence_steps"] or
            min(config["epochs"], config["minibatch_sequences"], config["checkpoint_every_seconds"],
                config.get("eval_interval_seconds", 8), config.get("burst_steps", 1),
                config.get("checkpoint_eval_every_seconds", 60)) <= 0
            or config.get("checkpoint_eval_ticks", 840) < 4
            or config.get("checkpoint_role", "mixed") not in SPECIALIST_TIERS
            or len(checkpoint_weights) != len(TIERS)
            or any(not 0 <= value < float("inf") for value in checkpoint_weights) or not sum(checkpoint_weights)
            or config.get("pose_aux") and any(not 0 < config.get(key, 0) < float("inf")
                                             for key in ("yaw_loss_weight", "lock_loss_weight"))
            or any(item.get("reverse_stage") not in (None, "lock", "branch", "micro_lock") for item in stages)
            or not 0 < config.get("clip_range", 0.2) < 1
            or config.get("min_steps_before_early_stop", 350_000) < 0 or config.get("episodes_per_wad", 1) < 1
            or not fractions or fractions[0] != 0 or fractions != sorted(set(fractions)) or fractions[-1] >= 1
            or any(len(x["weights"]) != 4 or min(x["weights"]) < 0 or not sum(x["weights"]) for x in stages)):
        parser.error("Invalid worker, rollout or curriculum configuration")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train(args, config, started)
    except Exception as error:
        atomic_save(args.out_dir / "train_summary.json", {
            "run_id": args.run_id, "status": "failed", "error": str(error),
            "wall_sec": time.monotonic() - started,
        }, json_file=True)
        raise


if __name__ == "__main__":
    main()
