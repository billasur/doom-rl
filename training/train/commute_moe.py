"""Fit only the policy gate and its value output with categorical PPO."""

import argparse
import json
import math
import os
import signal
import time
from functools import partial
from pathlib import Path

for thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_env, "1")

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from doom_env import action_space, observation_space
from maps.pool import V2_FREEDOOM_TRAIN, freedoom_wad, obsidian_maps
from models.commute_moe import FrozenExpertGate, TEMPERATURE_KEY
from models.v1_spatial_net import ACTION_TABLE
from train.train_v1 import CurriculumEnv, TIERS, isolated_vizdoom_worker


def load_expert(path, temperature):
    weights = torch.load(path, map_location="cpu", weights_only=True)
    embedded = weights.pop(TEMPERATURE_KEY, None)
    if temperature is None:
        if embedded is None:
            raise ValueError(f"Supply the certified temperature for {path}")
        temperature = embedded.item()
    elif embedded is not None and embedded.item() != temperature:
        raise ValueError(f"Temperature disagrees with the checkpoint metadata for {path}")
    return weights, temperature


def make_collectors(args, model, env_type=CurriculumEnv):
    maps = [tuple(sorted((args.maps.resolve() / tier).glob("*.wad"))) for tier in TIERS]
    if any(not group for group in maps):
        raise ValueError("Expected nonempty training WAD pools for all four procedural tiers")
    freedoom = freedoom_wad()
    groups = [maps[-1], tuple((freedoom, name) for name in V2_FREEDOOM_TRAIN)]
    if args.obsidian_train is None:raise ValueError("The balanced gate curriculum needs the OBSIDIAN training pool")
    groups.append(obsidian_maps(args.obsidian_train.resolve(),"train",compact_count=getattr(args,"obsidian_count",491)))
    sources=(groups,[.25,.375,.375])
    mix=dict(freedoom=.25,obsidian=.25,tier3=.25,tier4=.25)
    rewards=dict(key_pickup=10.,locked_open=15.,exit=20.,damage_dealt=.03,kill=2.5)
    worker = partial(env_type, maps, (1,1,1,1), args.episode_ticks,reward_settings=rewards,
                     tier4_sources=sources,generalist_sources=sources,reward_mode="v2_generalist",
                     source_names=("procedural","freedoom","obsidian"),fusion_episode_mix=mix,balanced_episode_mix=True,
                     freedoom_random_start=True,categorical=True,depth=model.depth,visits=model.visits)
    return gym.vector.AsyncVectorEnv(
        [worker] * args.workers, context="spawn", worker=isolated_vizdoom_worker,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


@torch.no_grad()
def collect(env, model, observation, recurrent, starts, length, action_table=ACTION_TABLE):
    device = next(model.gate.parameters()).device
    tensors = {name: [] for name in ("observation", "experts", "action", "log_prob", "value", "delta", "continues")}
    for _ in range(length):
        packed = torch.as_tensor(observation, device=device)
        expert_logs, recurrent = model.expert_log_probabilities(packed, recurrent, starts)
        logits, value = model.mixture(packed, expert_logs)
        distribution = Categorical(logits=logits)
        chosen = distribution.sample()
        observation, rewards, terminated, truncated, info = env.step(action_table[chosen.cpu().numpy()])
        done = terminated | truncated
        starts = torch.as_tensor(done, device=device)
        next_value = model.gate_outputs(torch.as_tensor(observation, device=device))[1]
        timeout = truncated & ~terminated
        if timeout.any():
            finals = np.stack(info["final_obs"][timeout])
            next_value[torch.as_tensor(timeout, device=device)] = model.gate_outputs(
                torch.as_tensor(finals, device=device))[1]
        next_value = next_value.masked_fill(torch.as_tensor(terminated, device=device), 0)
        delta = torch.as_tensor(rewards, dtype=torch.float32, device=device) + .995 * next_value - value
        gate_inputs = packed[:, :model.previous_column+1] if hasattr(model, "previous_column") else packed
        for name, item in zip(tensors, (gate_inputs.clone(), expert_logs, chosen, distribution.log_prob(chosen),
                                         value, delta, ~starts)):
            tensors[name].append(item)
    batch = {key: torch.stack(items) for key, items in tensors.items()}
    advantages = torch.empty_like(batch["delta"])
    previous = torch.zeros_like(advantages[0])
    for i in range(length - 1, -1, -1):
        previous = batch["delta"][i] + .995 * .95 * batch["continues"][i] * previous
        advantages[i] = previous
    batch["returns"] = advantages + batch["value"]
    batch["advantage"] = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return {key: item.flatten(0, 1) for key, item in batch.items()}, observation, recurrent, starts


def update(model, optimizer, batch):
    size = batch["action"].numel()
    device = batch["action"].device
    for _ in range(3):
        for indices in torch.randperm(size, device=device).split(128):
            logits, value = model.mixture(batch["observation"][indices], batch["experts"][indices])
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(batch["action"][indices]) - batch["log_prob"][indices]).exp()
            advantage = batch["advantage"][indices]
            policy = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
            value_loss = .5 * (value - batch["returns"][indices]).square().mean()
            loss = policy + .5 * value_loss - .002 * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gate.parameters(), .5)
            optimizer.step()


def atomic_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, allow_nan=False) + "\n")
    temporary.replace(path)


def export(model, optimizer, steps, updates, out):
    weights = model.checkpoint()
    for name in ("state.pt", "final_weights.pt"):
        path = out / name
        temporary = path.with_suffix(".tmp")
        payload=weights
        if name=="state.pt":
            from train.train_burst import cpu_tree
            payload=dict(model=weights,optimizer=cpu_tree(optimizer.state_dict()),global_step=steps,updates=updates)
        torch.save(payload, temporary)
        temporary.replace(path)


def train(args):
    began = time.monotonic()
    deadline = began + args.seconds - 5
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    a, temp_a = load_expert(args.expert_a, args.expert_a_temperature)
    b, temp_b = load_expert(args.expert_b, args.expert_b_temperature)
    experts, temperatures = [a,b], [temp_a,temp_b]
    gate_type, env_type, action_table = FrozenExpertGate, CurriculumEnv, ACTION_TABLE
    if getattr(args, "expert_c", None) is not None:
        from models.cross_gate import CrossGate, UNION
        from train.cross_curriculum import CrossCurriculum
        c,temp_c=load_expert(args.expert_c,args.expert_c_temperature)
        experts.append(c)
        temperatures.append(temp_c)
        gate_type,env_type,action_table=CrossGate,CrossCurriculum,UNION
    model = gate_type(observation_space(), action_space(), experts, temperatures).to(device)
    initial = [p.detach().clone() for p in model.gate.parameters()]
    initial_route = model.gate[-1].weight[:len(experts)].detach().clone()
    optimizer = torch.optim.Adam(model.gate.parameters(), lr=args.learning_rate, eps=1e-5)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cancelled = False

    def stop(*_):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    steps = updates = 0
    last_export = began
    env = make_collectors(args, model, env_type)
    try:
        observation, _ = env.reset(seed=args.seed)
        recurrent = starts = None
        while not cancelled and time.monotonic() < deadline and (args.max_steps is None or steps < args.max_steps):
            limit = args.rollout_steps if args.max_steps is None else min(
                args.rollout_steps, math.ceil((args.max_steps - steps) / args.workers))
            batch, observation, recurrent, starts = collect(env, model, observation, recurrent, starts, limit, action_table)
            if cancelled:
                break
            update(model, optimizer, batch)
            steps += batch["action"].numel()
            updates += 1
            now = time.monotonic()
            progress = dict(run_id=args.run_id, step=steps, global_step=steps, updated_at=time.time(),
                            steps_per_sec=steps / max(now - began, .001))
            atomic_report(args.out_dir / "live.json", progress)
            if updates == 1 or now - last_export >= 45:
                export(model, optimizer, steps, updates, args.out_dir)
                last_export = now
        if not updates:
            raise RuntimeError("Training ended before a complete PPO update")
        export(model, optimizer, steps, updates, args.out_dir)
        gate_delta = sum((p.detach() - old).square().sum().item()
                         for p, old in zip(model.gate.parameters(), initial)) ** .5
        route_delta = (model.gate[-1].weight[:len(experts)].detach() - initial_route).norm().item()
        frozen = all(torch.equal(value.detach().cpu(), original[key])
                     for expert, original in zip(model.experts, experts)
                     for key, value in expert.state_dict().items())
        if not frozen or not math.isfinite(gate_delta) or gate_delta == 0:
            raise RuntimeError("Expected frozen expert tensors and a changed gate")
        if not math.isfinite(route_delta) or route_delta == 0:
            raise RuntimeError("Expected nonzero update to the policy routing weights")
        counts=env.call("fusion_counts")
        report = dict(progress, steps=steps, updates=updates, frozen_experts_equal=frozen,
                      episode_counts={k:sum(c[k] for c in counts) for k in counts[0]},
                      gate_parameters=sum(p.numel() for p in model.gate.parameters()), gate_delta_l2=gate_delta,
                      route_delta_l2=route_delta, final_weights_path=str((args.out_dir / "final_weights.pt").resolve()))
        atomic_report(args.out_dir / "train_summary.json", report)
        print(json.dumps(report), flush=True)
    finally:
        env.close(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-a", type=Path, required=True)
    parser.add_argument("--expert-b", type=Path, required=True)
    parser.add_argument("--expert-a-temperature", type=float)
    parser.add_argument("--expert-b-temperature", type=float)
    parser.add_argument("--expert-c", type=Path)
    parser.add_argument("--expert-c-temperature", type=float)
    parser.add_argument("--obsidian-count", type=int, default=491)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--obsidian-train", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-id", default="commute_moe")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, default=90800)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episode-ticks", type=int, default=1120)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if not 5 < args.seconds < float("inf") or not 0 < args.learning_rate < float("inf") or min(
        args.workers, args.rollout_steps, args.episode_ticks // 4, args.max_steps if args.max_steps is not None else 1
    ) < 1:
        parser.error("Expected a bounded time budget and positive worker, rollout, episode and step settings")
    train(args)


if __name__ == "__main__":
    main()
