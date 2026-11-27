# Training

This folder holds the pipeline that produced the submitted agent, Extra 13: a learned router over
three frozen recurrent experts (R2_08, B08 and L4 student 2). The selected run is described in
[SELECTED_RUN.md](SELECTED_RUN.md) and [selected_run.json](selected_run.json).

Run every command from this directory. Dependencies are listed in [requirements.txt](requirements.txt);
all of them are already in the official image.

## Layout

| Path | Purpose |
| --- | --- |
| `models/` | Network definitions: `v0_net.py`, `v1_spatial_net.py` (spatial recurrent policy), `commute_moe.py`, `cross_gate.py`, `final_gate.py` (routers), `sanskar_policy.py` |
| `train/train_v0.py` | Train the v0 GRU policy with recurrent PPO on the template map |
| `train/train_v1.py` | Train the spatial policy with asymmetric recurrent PPO |
| `train/train_burst.py` | Train or resume a bounded v1 burst, optionally distilling frozen teachers |
| `train/universal_reward.py` | Training rewards and critic features from the engine state (training only) |
| `train/commute_moe.py` | Fit only the policy gate and its value output with categorical PPO |
| `train/final_gate.py` | Train the two- or three-expert gate used by the submission |
| `train/swa_merge.py` | Average 2 to 4 v1 checkpoints fine-tuned from a common initialization |
| `train/fusion.py`, `train/cross_curriculum.py`, `train/batched_rollout.py`, `train/checkpoint_eval.py` | Fusion schedules, collector, batched inference, development-map selection |
| `maps/gen_wads.py` | Generate deterministic UDMF training and held-out WADs |
| `maps/micro_drills.py` | Short lock exercises with real UDMF key pickups |
| `maps/pool.py` | Freedoom 2 train/eval split and OBSIDIAN map selection |
| `eval/eval_suite.py`, `eval/progression.py` | Held-out evaluation and progression plots by tier |
| `configs/v2_generalist.yaml` | PPO configuration for the generalist spatial policy |
| `submission/agent.py` | Training-side copy of the agent entry point |

## Pipeline

1. Maps: `python -m maps.gen_wads --split all` builds generated tiers. Freedoom 2 maps
   MAP01-07, MAP10 and MAP15 are held out for evaluation (`maps/pool.py`).
2. Experts: `python -m train.train_v1 --config configs/v2_generalist.yaml` trains the spatial
   policy (gamma 0.995, GAE lambda 0.95, clip 0.2, 32-step recurrent sequences, monster density 0.35).
   `train.train_burst` resumes and specializes checkpoints, including teacher distillation.
3. Router: `python -m train.final_gate --keycard ... --out-dir ...` freezes the experts and trains
   only the gate with PPO. See `--help` for all options.
4. Evaluation: `eval/eval_suite.py` scores checkpoints on held-out WADs.

Privileged engine state is used only for rewards and critic features during training. The submitted
agent reads only the official `doom_env` observation.
