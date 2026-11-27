"""Train the v0 GRU policy with recurrent PPO on the template map."""

import argparse
import os
import sys
import tempfile
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "maps" / "two_rooms.wad"
WEIGHTS_PATH = ROOT / "submission" / "v0_weights.pt"
EPISODE_TICKS = 2100
ROLLOUT_STEPS = 128
SEQUENCE_STEPS = 16
GAMMA = 0.99
GAE_LAMBDA = 0.95


def make_template_env(shaped=False):
    from doom_env import make_env

    env = make_env(str(MAP_PATH), "MAP01", episode_ticks=EPISODE_TICKS, render_buffers=False)
    if shaped:
        from train.universal_reward import UniversalReward

        return UniversalReward(env)
    return env


@dataclass
class Rollout:
    telemetry: torch.Tensor
    hidden: torch.Tensor
    starts: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def collect(envs, model, encode, obs, hidden, starts, episode_returns, recent_returns):
    workers = envs.num_envs
    shape = (ROLLOUT_STEPS, workers)
    telemetry = torch.empty((*shape, encode.size))
    states = torch.empty((*shape, model.hidden_size))
    episode_starts = torch.empty(shape, dtype=torch.bool)
    actions = torch.empty((*shape, envs.single_action_space.shape[0]))
    log_probs = torch.empty(shape)
    values = torch.empty(shape)
    rewards = torch.empty(shape)
    done_steps = torch.empty(shape, dtype=torch.bool)
    timeout_values = torch.zeros(shape)

    with torch.no_grad():
        for step in range(ROLLOUT_STEPS):
            telemetry[step] = torch.from_numpy(encode(obs))
            states[step] = hidden[0]
            episode_starts[step] = starts
            logits, value, hidden = model(telemetry[step : step + 1], hidden)
            policy = torch.distributions.Bernoulli(logits=logits[0])
            actions[step] = policy.sample()
            log_probs[step] = policy.log_prob(actions[step]).sum(-1)
            values[step] = value[0]
            obs, reward, terminated, truncated, info = envs.step(actions[step].numpy().astype(np.int8))
            done = terminated | truncated
            starts = torch.from_numpy(done)
            done_steps[step] = starts
            rewards[step] = torch.from_numpy(reward)
            episode_returns += reward
            recent_returns.extend(episode_returns[done].tolist())
            episode_returns[done] = 0

            # A time limit bootstraps from the final frame using the pre-reset GRU state.
            timed_out = np.flatnonzero(truncated & ~terminated)
            if timed_out.size:
                final = torch.from_numpy(np.stack([encode(info["final_obs"][i]) for i in timed_out]))
                _, final_value, _ = model(final[None], hidden[:, timed_out])
                timeout_values[step, timed_out] = final_value[0]
            hidden = hidden.masked_fill(starts[None, :, None], 0)

        _, next_value, _ = model(torch.from_numpy(encode(obs))[None], hidden)

    advantages = torch.empty_like(rewards)
    gae = torch.zeros(workers)
    next_value = next_value[0]
    for step in reversed(range(ROLLOUT_STEPS)):
        continues = ~done_steps[step]
        bootstrap = next_value * continues + timeout_values[step]
        delta = rewards[step] + GAMMA * bootstrap - values[step]
        gae = delta + GAMMA * GAE_LAMBDA * continues * gae
        advantages[step] = gae
        next_value = values[step]
    rollout = Rollout(telemetry, states, episode_starts, actions, log_probs, advantages, values + advantages)
    return rollout, obs, hidden, starts


def unroll(model, telemetry, hidden, starts):
    """Run contiguous GRU segments and clear state wherever a worker resets."""
    boundaries = starts[1:].any(dim=1).nonzero().flatten().add(1).tolist()
    logits, values = [], []
    for start, end in zip([0, *boundaries], [*boundaries, len(telemetry)]):
        hidden = hidden.masked_fill(starts[start][None, :, None], 0)
        segment_logits, segment_values, hidden = model(telemetry[start:end], hidden)
        logits.append(segment_logits)
        values.append(segment_values)
    return torch.cat(logits), torch.cat(values)


def update(model, optimizer, rollout, entropy_weight, minibatch_sequences=8):
    steps, workers = rollout.advantages.shape
    num_sequences = (steps // SEQUENCE_STEPS) * workers
    offsets = torch.arange(SEQUENCE_STEPS)[:, None]
    normalized = (rollout.advantages - rollout.advantages.mean()) / (
        rollout.advantages.std(unbiased=False) + 1e-8
    )
    for _ in range(4):
        for sequences in torch.randperm(num_sequences).split(minibatch_sequences):
            first = (sequences // workers) * SEQUENCE_STEPS
            envs = sequences % workers
            frames = first[None] + offsets
            index = (frames, envs[None])
            logits, values = unroll(
                model,
                rollout.telemetry[index],
                rollout.hidden[first, envs][None],
                rollout.starts[index],
            )
            policy = torch.distributions.Bernoulli(logits=logits)
            ratio = (policy.log_prob(rollout.actions[index]).sum(-1) - rollout.log_probs[index]).exp()
            advantages = normalized[index]
            policy_loss = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
            value_loss = (values - rollout.returns[index]).square().mean()
            entropy = policy.entropy().sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def evaluate(model, encode, seed, episodes=4):
    """Compare the same map seeds and action RNG streams before and after PPO."""
    env = make_template_env()
    results = []
    try:
        with torch.inference_mode():
            for episode_seed in range(seed, seed + episodes):
                obs, _ = env.reset(seed=episode_seed)
                rng = torch.Generator().manual_seed(episode_seed)
                hidden = None
                total = 0.0
                steps = 0
                cells = set()
                while True:
                    logits, _, hidden = model(torch.from_numpy(encode(obs))[None, None], hidden)
                    action = torch.bernoulli(logits[0, 0].sigmoid(), generator=rng)
                    obs, reward, terminated, truncated, _ = env.step(action.to(torch.int8).numpy())
                    total += reward
                    steps += 1
                    state = env.privileged_state()
                    cells.add((int(state["x"] // 64), int(state["y"] // 64)))
                    if terminated or truncated:
                        break
                results.append((total, state["exit_reached"], state["kills"], len(cells), steps))
    finally:
        env.close()
    return results


def report_evaluation(label, results):
    returns, exits, kills, explored, steps = zip(*results)
    print(
        f"{label} | mean reward {np.mean(returns):+.3f} | exits {sum(exits)}/{len(exits)}"
        f" | kills {sum(kills)} | cells {list(explored)} | steps {list(steps)}",
        flush=True,
    )
    # Favor finishing the real episode, then combat, progress and speed.
    return sum(exits), sum(returns), sum(explored), -sum(steps)


def save_weights(model, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temp = Path(file.name)
        try:
            torch.save({key: value.cpu() for key, value in model.state_dict().items()}, file)
            file.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shaped", action="store_true", help="reward route progress during training")
    parser.add_argument("--output", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--resume", type=Path, help="initialize from an existing checkpoint")
    args = parser.parse_args()
    if args.workers < 1 or not 0 < args.seconds < float("inf") or args.seed < 0:
        parser.error("--workers and --seconds must be positive, and --seed must be nonnegative")

    from models.v0_net import RecurrentActorCritic, TelemetryEncoder

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    envs = gym.vector.AsyncVectorEnv(
        [partial(make_template_env, shaped=args.shaped)] * args.workers,
        context="spawn",
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        encode = TelemetryEncoder(envs.single_observation_space)
        model = RecurrentActorCritic(encode.size, envs.single_action_space)
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
        evaluation_seed = args.seed + args.workers
        report_evaluation("Start", evaluate(model, encode, evaluation_seed))
        best_score = best_weights = None
        obs, _ = envs.reset(seed=args.seed)
        hidden = torch.zeros(1, args.workers, model.hidden_size)
        starts = torch.ones(args.workers, dtype=torch.bool)
        episode_returns = np.zeros(args.workers)
        recent_returns = deque(maxlen=32)
        steps = 0
        start = time.perf_counter()
        elapsed = next_log = 0.0
        next_eval = 25.0
        while elapsed < args.seconds:
            rollout, obs, hidden, starts = collect(
                envs, model, encode, obs, hidden, starts, episode_returns, recent_returns
            )
            entropy_weight = 0.004 if args.shaped else 0.01
            update(model, optimizer, rollout, entropy_weight)
            steps += ROLLOUT_STEPS * args.workers
            elapsed = time.perf_counter() - start
            if elapsed >= next_log or elapsed >= args.seconds:
                reward = f"{np.mean(recent_returns):+.3f}" if recent_returns else "pending episode completion"
                print(
                    f"{elapsed:5.1f}s | {steps:7,d} steps | {steps / elapsed:7.0f} FPS | mean reward {reward}",
                    flush=True,
                )
                next_log = elapsed + 5
            if args.shaped and elapsed >= next_eval and elapsed < args.seconds:
                score = report_evaluation(f"Eval at {elapsed:.0f}s", evaluate(model, encode, evaluation_seed))
                if best_score is None or score > best_score:
                    best_score = score
                    best_weights = deepcopy(model.state_dict())
                next_eval = elapsed + 25
    finally:
        envs.close()

    score = report_evaluation("End", evaluate(model, encode, evaluation_seed))
    if best_score is not None and best_score > score:
        model.load_state_dict(best_weights)
    save_weights(model, args.output)
    print(f"Saved CPU weights to {args.output}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
