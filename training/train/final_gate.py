"""Train a sharp two- or three-expert gate on the six training domains."""
import argparse
from functools import partial
import math
import os
from pathlib import Path
import signal
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from doom_env import action_space, observation_space
from doom_env.spaces import ENTITY_TYPES, RAY_HIT_TYPES
from maps.pool import V2_FREEDOOM_TRAIN, freedoom_wad, obsidian_maps
from models.cross_gate import UNION
from models.final_gate import ENTITY_START, FinalGate
from models.v1_spatial_net import ENTITIES, RAYS, TELEMETRY_SIZE
from train.commute_moe import atomic_report, export, load_expert
from train.cross_curriculum import CrossCurriculum
from train.train_v1 import TIERS, isolated_vizdoom_worker

GAMMA = .995
GAE = .95
LOCK_HIT = RAY_HIT_TYPES.index("locked_door")
ENEMY_HIT = RAY_HIT_TYPES.index("enemy")
ENEMY_TYPES = tuple(i for i, name in enumerate(ENTITY_TYPES) if name.startswith("enemy"))


def build_model(args):
    specs = []
    if args.combat is not None:
        specs.append((args.combat, args.combat_temperature))
    rgb_expert = args.real_expert is None
    real_path, real_temperature = ((args.sanskar, args.sanskar_temperature) if rgb_expert else
                                   (args.real_expert, args.real_expert_temperature))
    specs.extend(((args.keycard, args.keycard_temperature), (real_path, real_temperature)))
    experts, temperatures = zip(*(load_expert(path, temp) for path, temp in specs))
    model = FinalGate((observation_space(), action_space()), experts, temperatures,
                      args.gate_mode, args.gate_tau, args.gate_architecture, rgb_expert=rgb_expert)
    return model.to(torch.device(args.device)), experts


def collectors(args, model):
    directory = args.maps.resolve()
    maps = [tuple(sorted((directory / tier).glob("*.wad"))) for tier in TIERS]
    if not all(maps):
        raise ValueError("Every procedural training tier requires at least one WAD")
    two_rooms = directory.parent / "two_rooms.wad"
    if two_rooms.is_file():
        maps[1] = (*maps[1], two_rooms)
    freedoom = tuple((freedoom_wad(), name) for name in V2_FREEDOOM_TRAIN)
    obsidian = obsidian_maps(args.obsidian_train.resolve(), "train", compact_count=args.obsidian_count)
    sources = ([maps[-1], freedoom, obsidian], [1/3] * 3)
    mix = dict.fromkeys(("freedoom", "obsidian", "combat", "procedural", "tier3", "tier4"), 1/6)
    if args.keycard_share is not None:
        for domain in mix:
            mix[domain] = args.keycard_share if domain == "tier3" else (1 - args.keycard_share) / 5
    if args.real_share is not None:
        rest = (1 - args.keycard_share - args.real_share) / 3
        mix.update(freedoom=.65 * args.real_share, obsidian=.35 * args.real_share,
                   combat=rest, procedural=rest, tier4=rest, tier3=args.keycard_share)
    rewards = dict(key_pickup=10., locked_open=args.lock_reward, exit=20., damage_dealt=.03, kill=2.5)
    worker = partial(CrossCurriculum, maps, (0, 1, 0, 0), args.episode_ticks, reward_settings=rewards,
                     tier4_sources=sources, generalist_sources=sources, reward_mode="v2_generalist",
                     source_names=("procedural", "freedoom", "obsidian"), fusion_episode_mix=mix,
                     balanced_episode_mix=args.keycard_share is None, freedoom_random_start=True, categorical=True,
                     depth=True, visits=model.visits, label_teacher_domains=True)
    return gym.vector.AsyncVectorEnv([worker] * args.workers, context="spawn", worker=isolated_vizdoom_worker,
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def route_targets(packed, domains, home_count, rgb_expert=True):
    """Derive auxiliary labels from training domains and legal telemetry only."""
    hits = packed[..., RAYS:2 * RAYS]
    keys = packed[..., TELEMETRY_SIZE - 9:TELEMETRY_SIZE - 3].bool().any(-1)
    keycard = (domains == 2) | (domains == 4) | keys | (hits == LOCK_HIT).any(-1)
    labels = torch.full_like(domains, -1)
    real_index = home_count - int(not rgb_expert)
    if real_index == 2:
        classes = packed[..., ENTITY_START:ENTITY_START + ENTITIES]
        mask = packed[..., ENTITY_START + ENTITIES:ENTITY_START + 2 * ENTITIES].bool()
        enemy = (hits == ENEMY_HIT).any(-1)
        for kind in ENEMY_TYPES:
            enemy |= ((classes == kind) & mask).any(-1)
        combat = ((domains == 0) | (domains == 3) | (domains == 6)) & enemy
        labels.masked_fill_(combat, 0)
    labels.masked_fill_((domains == 1) | (domains == 5), real_index)
    return labels.masked_fill_(keycard, real_index - 1)


def distributions(model, packed, expert_probs, scores, option=None):
    routes = Categorical(logits=scores / model.routing_temperature)
    if model.mode == "hard":
        option = routes.sample() if option is None else option
        action_probs = expert_probs[torch.arange(len(option), device=option.device), option]
        option_log_prob = routes.log_prob(option)
    else:
        action_probs = (routes.probs[..., None] * expert_probs).sum(-2)
        option = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
        option_log_prob = torch.zeros_like(scores[..., 0])
    return Categorical(logits=model.action_logits(action_probs, packed)), routes, option, option_log_prob


@torch.no_grad()
def collect(env, model, state, length):
    observation, info, experts, context, starts = state
    device = model.head.weight.device
    trajectory = []
    for _ in range(length):
        packed = torch.as_tensor(observation, device=device)
        domains = torch.as_tensor(info["teacher_domain"], device=device)
        labels = route_targets(packed, domains, model.home_count, model.rgb_expert)
        context = (packed.new_zeros((len(packed), model.context_width)) if context is None
                   else context.masked_fill(starts[:, None], 0))
        before = context
        scores, value, context = model.gate_outputs(packed, before)
        expert_probs, experts = model.expert_log_probabilities(packed, experts, starts)
        policy, _, option, option_log = distributions(model, packed, expert_probs, scores)
        choice = policy.sample()
        logprob = option_log + policy.log_prob(choice)
        observation, rewards, terminated, truncated, info = env.step(UNION[choice.cpu().numpy()])
        done = terminated | truncated
        starts = torch.as_tensor(done, device=device)
        next_context = context.masked_fill(starts[:, None], 0)
        next_value = model.gate_outputs(torch.as_tensor(observation, device=device), next_context)[1]
        timeout = truncated & ~terminated
        if timeout.any():
            selector = torch.as_tensor(timeout, device=device)
            finals = torch.as_tensor(np.stack(info["final_obs"][timeout]), device=device)
            next_value[selector] = model.gate_outputs(finals, context[selector])[1]
        next_value.masked_fill_(torch.as_tensor(terminated, device=device), 0)
        delta = torch.as_tensor(rewards, dtype=torch.float32, device=device) + GAMMA * next_value - value
        trajectory.append((packed[..., :model.previous_column + 1].clone(), before, expert_probs, labels,
                           option, choice, logprob, value, delta, ~starts))
    names = ("packed", "context", "experts", "labels", "option", "choice", "old_log", "value", "delta", "continues")
    buffer = dict(zip(names, map(torch.stack, zip(*trajectory))))
    advantage = torch.empty_like(buffer["delta"])
    tail = torch.zeros_like(advantage[0])
    for i in range(length - 1, -1, -1):
        tail = buffer["delta"][i] + GAMMA * GAE * buffer["continues"][i] * tail
        advantage[i] = tail
    buffer["returns"] = advantage + buffer["value"]
    buffer["advantage"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    batch = {key: tensor.flatten(0, 1) for key, tensor in buffer.items()}
    return batch, (observation, info, experts, context, starts)


def update(model, optimizer, batch, aux_weight, anchor=None, anchor_weight=0.):
    size = len(batch["choice"])
    device = model.head.weight.device
    if anchor is not None:
        with torch.no_grad():
            anchor_scores = anchor.gate_outputs(batch["packed"], batch["context"])[0]
            anchor_probs = (anchor_scores / anchor.routing_temperature).softmax(-1)
    for _ in range(3):
        for indices in torch.randperm(size, device=device).split(128):
            row = {key: value[indices] for key, value in batch.items()}
            scores, value, _ = model.gate_outputs(row["packed"], row["context"])
            actions, routes, _, option_log = distributions(model, row["packed"], row["experts"], scores, row["option"])
            ratio = (option_log + actions.log_prob(row["choice"]) - row["old_log"]).exp()
            policy = -torch.minimum(ratio * row["advantage"], ratio.clamp(.8, 1.2) * row["advantage"]).mean()
            critic = .25 * (value - row["returns"]).square().mean()
            aux = F.cross_entropy(scores, row["labels"], ignore_index=-1, reduction="sum")
            aux = aux / (row["labels"] >= 0).sum().clamp_min(1)
            loss = policy + critic + .002 * routes.entropy().mean() - .002 * actions.entropy().mean() + aux_weight * aux
            if anchor is not None:
                # Preserve the prior router on training steps without a keycard-specific label.
                protected = row["labels"] != model.home_count - 1 - int(not model.rgb_expert)
                kl = F.kl_div(routes.logits, anchor_probs[indices], reduction="none").sum(-1)
                loss = loss + anchor_weight * (kl * protected).sum() / protected.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], .5)
            optimizer.step()


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model, original = build_model(args)
    initial = model.head.weight.detach().clone()
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, eps=1e-5)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.init_only:
        export(model, optimizer, 0, 0, args.out_dir)
        return
    steps, updates = 0, 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict({name: checkpoint["model"][name] for name in model.state_dict()})
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.reset_learning_rate:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        steps, updates = checkpoint["global_step"], checkpoint["updates"]
        torch.manual_seed(args.seed + updates)
    anchor = None
    if args.anchor_weight:
        import copy
        anchor = copy.deepcopy(model).requires_grad_(False).eval()
    segment_start_step = steps
    started = time.monotonic()
    duration = args.seconds - args.save_reserve
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = collectors(args, model)
    try:
        observation, info = env.reset(seed=args.seed + updates)
        state = observation, info, None, None, None
        export_at = started
        while not stopped and (elapsed := time.monotonic() - started) < duration:
            if args.max_steps is not None and steps >= args.max_steps:
                break
            batch, state = collect(env, model, state, args.rollout_steps)
            coefficient = args.aux_weight * max(0.0, 1.0 - (args.elapsed_before + elapsed) / (.60 * (args.elapsed_before + args.seconds)))
            coefficient = max(coefficient, args.aux_floor)
            update(model, optimizer, batch, coefficient, anchor, args.anchor_weight)
            steps += len(batch["choice"])
            updates += 1
            now = time.monotonic()
            progress = dict(run_id=args.run_id, step=steps, global_step=steps, updated_at=time.time(),
                            steps_per_sec=(steps - segment_start_step) / max(.001, now - started), aux_weight=coefficient)
            atomic_report(args.out_dir / "live.json", progress)
            if updates == 1 or now - export_at >= 40:
                export(model, optimizer, steps, updates, args.out_dir)
                export_at = now
        if not updates:
            raise RuntimeError("Expected a completed PPO update")
        route_delta = (model.head.weight.detach()[:-1] - initial[:-1]).norm().item()
        frozen = all(torch.equal(value.detach().cpu(), weights[key])
                     for expert, weights in zip(model.experts, original, strict=True)
                     for key, value in expert.state_dict().items())
        if not frozen or not math.isfinite(route_delta) or route_delta <= 0:
            raise RuntimeError("Expected bit-identical frozen experts and changed routing weights")
        export(model, optimizer, steps, updates, args.out_dir)
        counts = env.call("fusion_counts")
        report = dict(progress, updates=updates, steps=steps, frozen_experts_equal=frozen,
                      route_delta_l2=route_delta, gate_mode=model.mode, gate_tau=model.tau,
                      gate_architecture=model.architecture, expert_count=len(model.experts),
                      episode_counts={domain: sum(c[domain] for c in counts) for domain in counts[0]},
                      final_weights_path=str((args.out_dir / "final_weights.pt").resolve()))
        atomic_report(args.out_dir / "train_summary.json", report)
        print(report, flush=True)
    finally:
        env.close(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("sanskar", "keycard", "combat"):
        parser.add_argument(f"--{role}", type=Path, required=role == "keycard")
        parser.add_argument(f"--{role}-temperature", type=float)
    parser.add_argument("--real-expert", type=Path)
    parser.add_argument("--real-expert-temperature", type=float)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gate-mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--gate-tau", type=float, choices=(.20, .35), default=.20)
    parser.add_argument("--gate-architecture", choices=("gru", "mlp"), default="gru")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--elapsed-before", type=float, default=0.)
    parser.add_argument("--reset-learning-rate", action="store_true")
    parser.add_argument("--anchor-weight", type=float, default=0.)
    parser.add_argument("--aux-floor", type=float, default=0.)
    parser.add_argument("--keycard-share", type=float)
    parser.add_argument("--real-share", type=float)
    parser.add_argument("--lock-reward", type=float, default=15.)
    parser.add_argument("--maps", type=Path)
    parser.add_argument("--obsidian-train", type=Path)
    parser.add_argument("--obsidian-count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seconds", type=float, default=2400)
    parser.add_argument("--save-reserve", type=float, default=120)
    parser.add_argument("--run-id", default="final_gate")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--aux-weight", type=float, default=.50)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--episode-ticks", type=int, default=1120)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if (args.sanskar is None) == (args.real_expert is None):
        parser.error("Specify exactly one real-map expert: --sanskar or --real-expert")
    if not args.init_only and (args.maps is None or args.obsidian_train is None or not 0 <= args.save_reserve < args.seconds
                               or min(args.workers, args.rollout_steps, args.episode_ticks // 4, args.obsidian_count) < 1):
        parser.error("Training requires both WAD pools, positive dimensions, and time above the save reserve")
    if not (0 < args.learning_rate < float("inf") and 0 <= args.aux_weight <= 1):
        parser.error("Invalid learning rate or routing supervision weight")
    if not (0 <= args.aux_floor <= args.aux_weight and args.anchor_weight >= 0
            and (args.keycard_share is None or 0 < args.keycard_share < 1)):
        parser.error("Invalid polish mix or anchor coefficients")
    if args.real_share is not None and (args.keycard_share is None or not 0 < args.real_share < 1 - args.keycard_share):
        parser.error("--real-share requires a positive keycard share and room for other domains")
    train(args)


if __name__ == "__main__":
    main()
