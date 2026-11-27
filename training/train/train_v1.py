"""Train the spatial DOOM policy with asymmetric recurrent PPO."""

import argparse
import os
import sys
import tempfile
import time
from collections import deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import gymnasium as gym
import numpy as np
import torch
import yaml
from doom_env import action_space, make_env, observation_space
from gymnasium.vector.async_vector_env import _async_worker

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from models.v1_spatial_net import (
    ACTION_TABLE, RAYS, TELEMETRY_SIZE, USE_ACTIONS, ActionHistory, SpatialActorCritic,
    SpatialState, mask_use, pack_policy, policy_width,
)
from train.fusion import EpisodeMix
from train.train_v0 import save_weights
from train.universal_reward import CRITIC_SIZE, UniversalReward

TIERS = ("tier1_combat", "tier2_explore", "tier3_keycard", "tier4_gauntlet")


def isolated_vizdoom_worker(*args, **kwargs):
    """Keep ViZDoom's relative runtime files private for each spawned worker."""
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="doom-worker-") as directory:
        os.chdir(directory)
        try:
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[name] = "1"
            torch.set_num_threads(1)
            return _async_worker(*args, **kwargs)
        finally:
            os.chdir(original)


class CompactTelemetry(gym.ObservationWrapper):
    def __init__(self, env, action_history=False, depth=False, visits=False, categorical=False):
        super().__init__(env)
        self.action_history = action_history
        self.depth = depth
        self.visits = visits
        self.categorical = categorical
        self.history = None
        width = policy_width(action_history=action_history, depth=depth, visits=visits, categorical=categorical)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (width,), np.float32)

    def reset(self, **kwargs):
        self.history = (ActionHistory(action_history=self.action_history, visits=self.visits, categorical=self.categorical)
                        if self.action_history or self.visits or self.categorical else None)
        return super().reset(**kwargs)

    def step(self, action):
        if self.history is not None:
            self.history.action = (int(np.flatnonzero((ACTION_TABLE == action).all(axis=1))[0]) if self.categorical
                                   else np.asarray(action, dtype=np.int64))
        return super().step(action)

    def observation(self, observation):
        return pack_policy(observation, self.history, self.depth)


class CurriculumEnv(gym.Env):
    def __init__(self, map_groups, weights, episode_ticks, episodes_per_wad=1, reward_settings=None, *,
                 warmup_maps=(), warmup=False, tier4_sources=(), tier4_reward=None,
                 reverse_maps=None, reverse_stage=None, combat_maps=(), combat_drill=False,
                 explore_maps=(), action_history=False, pose_aux=False, depth=False, visits=False,
                 reward_mode=None, movement_penalty=False, categorical=False,
                 generalist_sources=None, keycard_drill_prob=0.0, branch_drills_only=False, freedoom_random_start=False,
                 label_teacher_domains=False, fusion_episode_mix=None, source_names=(), balanced_episode_mix=False):
        width = policy_width(action_history=action_history, depth=depth, visits=visits, categorical=categorical)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (width,), np.float32)
        self.action_space = action_space()
        self.map_groups = map_groups
        self.episode_ticks = episode_ticks
        self.episodes_per_wad = episodes_per_wad
        self.reward_settings = reward_settings
        self.reward_mode = reward_mode
        self.movement_penalty = movement_penalty
        self.tier4_reward = {**(reward_settings or {}), **(tier4_reward or {})}
        self.warmup_maps = warmup_maps
        self.reverse_maps = reverse_maps or {}
        self.fusion_mix = EpisodeMix(fusion_episode_mix, source_names, balanced_episode_mix) if fusion_episode_mix is not None else None
        self.generalist_sources = generalist_sources
        self.label_teacher_domains = label_teacher_domains
        self.keycard_drill_prob = keycard_drill_prob
        self.branch_drills_only = branch_drills_only
        self.freedoom_random_start = freedoom_random_start
        self.combat_maps = combat_maps
        self.explore_maps = explore_maps
        self.action_history = action_history
        self.pose_aux = pose_aux
        self.depth = depth
        self.visits = visits
        self.categorical = categorical
        self.drill_episodes = np.zeros(4, dtype=np.int64)
        self.tier4_sources = tier4_sources
        self.wad_loads = self.episodes_started = self.warmup_episodes = 0
        self.tier4_source_episodes = np.zeros(len(tier4_sources[0]) if tier4_sources else 0, dtype=np.int64)
        self.env = self.map_path = None
        self.set_stage(weights, warmup, reverse_stage, combat_drill)

    def set_stage(self, weights, warmup=False, reverse_stage=None, combat_drill=False):
        self.probabilities = np.asarray(weights, dtype=np.float64) / sum(weights)
        self.warmup = warmup
        self.reverse_stage = reverse_stage
        self.combat_drill = combat_drill
        self.map_episodes = self.episodes_per_wad

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self.env is None or self.map_episodes >= self.episodes_per_wad:
            self.tier = self.np_random.choice(len(self.map_groups), p=self.probabilities)
            maps = self.map_groups[self.tier]
            self.tier4_source = None
            self.drill = None
            reverse = self.reverse_stage if self.tier == 2 else None
            if self.fusion_mix is not None:
                bucket = self.fusion_mix.draw(self.np_random)
                reverse = None
                if bucket == "keycard":
                    self.tier, self.drill, reverse = 2, 1, "branch"
                    maps = self.reverse_maps[reverse]
                elif bucket in ("combat", "tier3", "tier4"):
                    self.tier = {"combat":0,"tier3":2,"tier4":3}[bucket]
                    maps = self.map_groups[self.tier]
                elif bucket in self.fusion_mix.external:
                    self.tier4_source = self.fusion_mix.external[bucket]
                    self.tier, maps = 3, self.generalist_sources[0][self.tier4_source]
                # The procedural bucket keeps the ordinary T1–T4 draw above.
            elif self.generalist_sources:
                reverse = None
                if self.keycard_drill_prob and self.np_random.random() < self.keycard_drill_prob:
                    self.tier = 2
                    self.drill = 1 if self.branch_drills_only else int(self.np_random.integers(2))
                    reverse = ("micro_lock", "branch")[self.drill]
                    maps = self.reverse_maps[reverse]
                else:
                    sources, probabilities = self.generalist_sources
                    self.tier4_source = self.np_random.choice(len(sources), p=probabilities)
                    if self.tier4_source:
                        self.tier, maps = 3, sources[self.tier4_source]
            elif self.tier == 1 and self.explore_maps and self.np_random.random() < 0.35:
                maps, self.drill = self.explore_maps, 3
            elif reverse:
                maps = self.reverse_maps[reverse]
                self.drill = 0 if reverse in ("lock", "micro_lock") else 1
            elif self.tier == 3:
                if self.warmup:
                    maps = self.warmup_maps
                elif self.combat_drill:
                    maps, self.drill = self.combat_maps, 2
                elif self.tier4_sources:
                    sources, probabilities = self.tier4_sources
                    self.tier4_source = self.np_random.choice(len(sources), p=probabilities)
                    maps = sources[self.tier4_source]
            choices = maps if seed is not None else [path for path in maps if path != self.map_path]
            choices = choices or maps
            path = choices[self.np_random.integers(len(choices))]
            if self.map_path != path:
                self.close()
                wad, map_name = path if isinstance(path, tuple) else (path, "MAP01")
                ticks = min(self.episode_ticks, {"lock": 560, "micro_lock": 80, "branch": 1120}.get(reverse, self.episode_ticks))
                self.env = CompactTelemetry(UniversalReward(
                    make_env(wad, map_name, episode_ticks=ticks, render_buffers=self.depth),
                    self.tier4_reward if self.tier == 3 else self.reward_settings,
                    reverse_stage=reverse, auxiliary=self.pose_aux, reward_mode=self.reward_mode,
                    movement_penalty=self.movement_penalty,
                    freedoom_random_start=self.freedoom_random_start and Path(wad).name == "freedoom2.wad",
                ), action_history=self.action_history, depth=self.depth, visits=self.visits, categorical=self.categorical)
                self.map_path = path
                self.wad_loads += 1
            self.map_episodes = 0
        self.map_episodes += 1
        self.episodes_started += 1
        self.warmup_episodes += int(self.tier == 3 and self.warmup)
        if self.drill is not None:
            self.drill_episodes[self.drill] += 1
        if self.tier4_source is not None:
            self.tier4_source_episodes[self.tier4_source] += 1
        obs, info = self.env.reset(seed=int(self.np_random.integers(2**31)), options=options)
        return obs, self.annotate(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self.annotate(info)

    def fusion_counts(self):
        return self.fusion_mix.counts.copy() if self.fusion_mix is not None else {}

    def annotate(self, info):
        if self.label_teacher_domains:
            domain = int(self.tier)
            if self.tier4_source:
                wad = self.map_path[0] if isinstance(self.map_path, tuple) else self.map_path
                domain = 5 if Path(wad).name == "freedoom2.wad" else 6
            elif domain == 2 and info["keys_picked"]:
                domain = 4
            info["teacher_domain"] = domain
        return dict(info, tier=int(self.tier), wad_loads=self.wad_loads, episodes_started=self.episodes_started,
                    warmup_episodes=self.warmup_episodes, tier4_source_episodes=self.tier4_source_episodes.copy(),
                    drill_episodes=self.drill_episodes.copy())

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = self.map_path = None


def tensor(array, device):
    item = torch.as_tensor(array)
    return item.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else item


def precision(device, config):
    return torch.amp.autocast("cuda", dtype=config["amp_dtype"]) if "amp_dtype" in config else nullcontext()


def policy_inputs(obs, privileged, device):
    if device.type == "cuda":
        packed = tensor(np.concatenate((obs, privileged), axis=-1), device)
        return packed[:, :obs.shape[-1]], packed[:, obs.shape[-1]:]
    return tensor(obs, device), tensor(privileged, device)


def adverse_transition(before, after, done):
    """Use the outcome of the action, including the outgoing episode in SAME_STEP mode."""
    def outcome(key):
        current = after[key]
        if not done.any():
            return current
        return np.where(done, after["final_info"][key], current)

    return (outcome("dead") | (outcome("damage_taken") - before["damage_taken"] >= 10)
            | (outcome("stuck_steps") > before["stuck_steps"]))


@torch.no_grad()
def collect(envs, model, obs, info, state, starts, config, device, stats):
    batch = {key: [] for key in ("obs", "privileged", "starts", "actions", "log_probs", "values",
                                 "rewards", "continues", "timeouts", "next_rays", "tier")}
    pose_aux = model.lock_head is not None
    if pose_aux:
        batch["auxiliary_targets"] = []
    if model.categorical:
        batch["prev_used"] = []
    guided = "teacher_domain_weights" in config
    if guided:
        batch.update(teacher_domain=[], teacher_bad_transition=[], teacher_key_held=[])
    states = []
    for _ in range(config["rollout_steps"]):
        telemetry, privileged = policy_inputs(obs, info["privileged"], device)
        states.append(state)
        with precision(device, config):
            logits, value, _, state = model(telemetry, state, privileged)
        logits, value = logits.float(), value.float()
        if model.categorical:
            first, last = USE_ACTIONS
            prev_used = telemetry[:, TELEMETRY_SIZE + first:TELEMETRY_SIZE + last + 1].bool().any(-1)
            batch["prev_used"].append(prev_used)
            policy = torch.distributions.Categorical(logits=mask_use(logits, prev_used), validate_args=False)
            action = policy.sample()
            buttons = model.action_table[action]
            log_prob = policy.log_prob(action)
        else:
            policy = torch.distributions.Bernoulli(logits=logits, validate_args=False)
            action = policy.sample()
            buttons = action.to(torch.int8)
            log_prob = policy.log_prob(action).sum(-1)
        tiers = info["tier"]
        if guided:
            before = info
            batch["teacher_domain"].append(torch.from_numpy(info["teacher_domain"]))
            batch["teacher_key_held"].append(torch.from_numpy(info["keys_picked"] > 0))
        if pose_aux:
            batch["auxiliary_targets"].append(torch.from_numpy(info["auxiliary_targets"]))
        stats["tiers"] += np.bincount(tiers, minlength=len(TIERS))
        envs.step_async(buttons.cpu().numpy())
        obs, rewards, terminated, truncated, info = envs.step_wait()
        done = terminated | truncated
        if guided:
            batch["teacher_bad_transition"].append(torch.from_numpy(adverse_transition(before, info, done)))
        next_starts = tensor(done, device)
        timeouts = torch.zeros_like(value)
        timed_out = np.flatnonzero(truncated & ~terminated)
        if timed_out.size:
            indices = tensor(timed_out, device)
            final, final_privileged = policy_inputs(
                np.stack(info["final_obs"][timed_out]), info["final_info"]["privileged"][timed_out], device
            )
            with precision(device, config):
                _, bootstraps, _, _ = model(final, SpatialState(*(item[indices] for item in state)), final_privileged)
            timeouts[indices] = bootstraps.float()
        rows = (telemetry, privileged, starts, action, log_prob, value,
                torch.as_tensor(rewards, dtype=torch.float32), ~next_starts, timeouts,
                torch.from_numpy(obs[:, :RAYS]), torch.from_numpy(tiers))
        for key, item in zip(batch, rows):
            batch[key].append(item)
        stats["returns"] += rewards
        if done.any():
            stats["recent"].extend(stats["returns"][done].tolist())
            stats["exits"].extend(info["final_info"]["exit_reached"][done].tolist())
            if "cells" in stats:
                stats["cells"].extend(info["final_info"]["cells_visited"][done].tolist())
            stats["returns"][done] = 0
        starts = next_starts
        state = state.reset_where(starts)

    with precision(device, config):
        telemetry, privileged = policy_inputs(obs, info["privileged"], device)
        _, next_value, _, _ = model(telemetry, state, privileged)
    next_value = next_value.float()
    batch = {key: torch.stack(rows) for key, rows in batch.items()}
    if device.type == "cuda":
        for key in ("rewards", "next_rays", "tier"):
            batch[key] = tensor(batch[key], device)
        if guided:
            for key in ("teacher_domain", "teacher_bad_transition", "teacher_key_held"):
                batch[key] = tensor(batch[key], device)
        if pose_aux:
            batch["auxiliary_targets"] = tensor(batch["auxiliary_targets"], device)
    batch["states"] = SpatialState(*(torch.stack(field) for field in zip(*states)))
    gae = torch.zeros_like(next_value)
    advantages = torch.empty_like(batch["values"])
    gamma = config["gamma"]
    for step in reversed(range(config["rollout_steps"])):
        continues = batch["continues"][step]
        delta = (batch["rewards"][step] + gamma * (next_value * continues + batch["timeouts"][step])
                 - batch["values"][step])
        gae = delta + gamma * config["gae_lambda"] * continues * gae
        advantages[step] = gae
        next_value = batch["values"][step]
    batch["returns"] = advantages + batch["values"]
    batch["advantages"] = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return batch, obs, info, state, starts


def ppo_update(model, optimizer, batch, config, device, *, scaler=None, kl_weight=0, profile=None):
    steps, workers = batch["values"].shape
    length = config["sequence_steps"]
    sequence_count = (steps // length) * workers
    offsets = torch.arange(length, device=device)[:, None]
    auxiliary_losses, kl_losses = [], []
    pose_losses = []
    for _ in range(config["epochs"]):
        for sequences in torch.randperm(sequence_count, device=device).split(config["minibatch_sequences"]):
            first, envs = (sequences // workers) * length, sequences % workers
            index = first[None] + offsets, envs[None]
            telemetry, privileged, starts = (batch[key][index] for key in ("obs", "privileged", "starts"))
            state = SpatialState(*(item[first, envs] for item in batch["states"]))
            with precision(device, config):
                outputs = model.sequence(telemetry, state, starts, privileged, predict_pose="auxiliary_targets" in batch)
                logits, values, predictions = outputs[:3]
                if profile is not None and not profile:
                    profile.update(autocast=torch.is_autocast_enabled(device.type),
                                   forward_logits_dtype=str(logits.dtype), grad_scaler=scaler is not None)
            actions = batch["actions"][index]
            if model.categorical:
                masked = mask_use(logits, batch["prev_used"][index])
                policy = torch.distributions.Categorical(logits=masked, validate_args=False)
                log_prob = policy.log_prob(actions)
                entropy = policy.entropy().mean()
            else:
                policy = torch.distributions.Bernoulli(logits=logits.float(), validate_args=False)
                log_prob = policy.log_prob(actions).sum(-1)
                entropy = policy.entropy().sum(-1).mean()
            ratio = (log_prob - batch["log_probs"][index]).exp()
            advantage = batch["advantages"][index]
            clip = config.get("clip_range", 0.2)
            actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
            critic_loss = (values.float() - batch["returns"][index]).square().mean()
            valid = batch["continues"][index]
            ray_error = (predictions.float() - batch["next_rays"][index]).square().mean(-1)
            auxiliary_loss = (ray_error * valid).sum() / valid.sum().clamp_min(1)
            loss = actor_loss + 0.5 * critic_loss - config["entropy_weight"] * entropy
            loss = loss + config["ray_loss_weight"] * auxiliary_loss
            if "auxiliary_targets" in batch:
                yaw, lock = outputs[3:]
                expected_yaw, sighted, forward, right = batch["auxiliary_targets"][index].unbind(-1)
                yaw_error = torch.nn.functional.smooth_l1_loss(yaw.float(), expected_yaw, beta=0.1, reduction="none")
                yaw_loss = (yaw_error * ~starts).sum() / (~starts).sum().clamp_min(1)
                sight_loss = torch.nn.functional.binary_cross_entropy_with_logits(lock[..., 0].float(), sighted)
                offsets_error = (lock[..., 1:].float() - torch.stack((forward, right), dim=-1)).square().mean(-1)
                coordinate_loss = (offsets_error * sighted).sum() / sighted.sum().clamp_min(1)
                loss = loss + config["yaw_loss_weight"] * yaw_loss + config["lock_loss_weight"] * (
                    sight_loss + coordinate_loss)
                pose_losses.append(torch.stack((yaw_loss, sight_loss, coordinate_loss)).detach())
            if "teacher_probs" in batch:
                target = batch["teacher_probs"][index]
                if model.categorical:
                    per_step = (torch.special.xlogy(target, target) - target * policy.logits).sum(-1)
                    if "teacher_valid" in batch:
                        eligible = batch["teacher_valid"][index]
                        kl = (per_step * eligible).sum() / eligible.sum().clamp_min(1)
                    else:
                        kl = per_step.mean()
                else:
                    teacher = torch.distributions.Bernoulli(probs=target.clamp(1e-6, 1-1e-6), validate_args=False)
                    kl = torch.distributions.kl_divergence(teacher, policy).sum(-1).mean()
                loss = loss + kl_weight * kl
                kl_losses.append(kl.detach())
            if profile is not None and "loss_dtype" not in profile:
                profile["loss_dtype"] = str(loss.dtype)
            if model.categorical:
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite categorical PPO loss")
                if profile is not None and "first_loss" not in profile:
                    profile.update(first_loss=loss.detach().item(), actions_dtype=str(actions.dtype),
                                   action_shape=list(batch["actions"].shape))
            optimizer.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            auxiliary_losses.append(auxiliary_loss.detach())
    ray_loss = torch.stack(auxiliary_losses).mean().item()
    if profile is not None and pose_losses:
        profile["pose_losses"] = dict(zip(("yaw", "locked_sighting", "locked_offsets"),
                                          torch.stack(pose_losses).mean(dim=0).tolist()))
    return (ray_loss, torch.stack(kl_losses).mean().item()) if kl_losses else ray_loss


def train(args, config):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    map_groups = [tuple(sorted((args.maps / tier).resolve().glob("*.wad"))) for tier in TIERS]
    if any(not group for group in map_groups):
        raise ValueError(f"Expected a nonempty directory for every tier under {args.maps}")
    stages = config["stages"]
    make_worker = partial(CurriculumEnv, map_groups, stages[0]["weights"], config["episode_ticks"])
    envs = gym.vector.AsyncVectorEnv(
        [make_worker] * config["workers"], context="spawn", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        worker=isolated_vizdoom_worker,
    )
    try:
        model = SpatialActorCritic(observation_space(), envs.single_action_space).to(device)
        if model.privileged_size != CRITIC_SIZE:
            raise ValueError("Reward wrapper and critic disagree on feature count")
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-5)
        obs, info = envs.reset(seed=config["seed"])
        state = model.initial_state(tensor(obs, device))
        starts = torch.ones(config["workers"], dtype=torch.bool, device=device)
        stats = {"returns": np.zeros(config["workers"]), "recent": deque(maxlen=100),
                 "exits": deque(maxlen=100), "tiers": np.zeros(len(TIERS), dtype=np.int64)}
        start = time.monotonic()
        steps = stage = 0
        elapsed = next_report = 0.0
        next_save = config["checkpoint_every_seconds"]
        print(f"Training on {device} with {config['workers']} workers and {[len(x) for x in map_groups]} maps", flush=True)
        while elapsed < config["seconds"]:
            if stage + 1 < len(stages) and elapsed >= stages[stage + 1]["start"] * config["seconds"]:
                stage += 1
                envs.call("set_stage", stages[stage]["weights"])
                obs, info = envs.reset()
                starts.fill_(True)
                state = model.initial_state(tensor(obs, device))
                stats["returns"].fill(0)
                print(f"Stage {stage + 1} | tier weights {stages[stage]['weights']}", flush=True)
            batch, obs, info, state, starts = collect(envs, model, obs, info, state, starts, config, device, stats)
            ray_loss = ppo_update(model, optimizer, batch, config, device)
            steps += config["workers"] * config["rollout_steps"]
            elapsed = time.monotonic() - start
            if elapsed >= next_report or elapsed >= config["seconds"]:
                reward = f"{np.mean(stats['recent']):+.2f}" if stats["recent"] else "pending"
                print(f"{elapsed:6.1f}s | {steps:8,d} steps | {steps / elapsed:.0f} FPS | "
                      f"return {reward} | exits {sum(stats['exits'])}/{len(stats['exits'])} | "
                      f"ray MSE {ray_loss:.4f} | tier steps {stats['tiers'].tolist()}", flush=True)
                next_report = elapsed + 10
            if elapsed >= next_save:
                checkpoint = args.output.with_name(f"{args.output.stem}_{steps:09d}.pt")
                save_weights(model, checkpoint)
                print(f"Saved {checkpoint}", flush=True)
                next_save += config["checkpoint_every_seconds"]
    finally:
        envs.close()
    save_weights(model, args.output)
    print(f"Saved CPU weights to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v1.yaml")
    parser.add_argument("--maps", type=Path, default=ROOT / "maps" / "train")
    parser.add_argument("--output", type=Path, default=ROOT / "submission" / "v1_weights.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for key in ("seconds", "workers", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if (not 0 < config["seconds"] < float("inf") or config["workers"] < 1 or config["seed"] < 0
            or config["sequence_steps"] < 1 or config["rollout_steps"] < 1
            or config["rollout_steps"] % config["sequence_steps"]
            or min(config["minibatch_sequences"], config["epochs"], config["checkpoint_every_seconds"]) < 1):
        parser.error("Invalid duration, worker count, seed or rollout/sequence configuration")
    stages = config["stages"]
    starts = [stage["start"] for stage in stages]
    if (not starts or starts[0] != 0 or starts != sorted(set(starts)) or starts[-1] >= 1
            or any(len(x["weights"]) != len(TIERS) or min(x["weights"]) < 0 or not sum(x["weights"]) for x in stages)):
        parser.error("Invalid curriculum start fractions or tier weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train(args, config)


if __name__ == "__main__":
    main()
