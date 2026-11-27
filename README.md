# Extra 13 · Can It Play DOOM?

DevForge Track 1 submission by Sanskar Kharya, Prudhvi Raj, and Prahlad.

Extra 13 learns when to use three frozen recurrent specialists for real-map exploration, keycards, and combat. Each expert maintains its own GRU state and 16×16 egocentric grid. All three update on every decision so the router can switch experts without losing their history. The submitted policy uses official telemetry and an 8×32 pooled depth strip.

[Project brief](brief.pdf) · [Agent](submission/agent.py) · [Git LFS weights](submission/v2_weights.pt) · [Training recipe](training/SELECTED_RUN.md)

## Validate the submission

These commands require Git LFS, uv, and Python 3.13.

```bash
git lfs pull
git lfs ls-files
uv run --with 'torch==2.15.0' python validate.py --device cpu
```

The starter's local environment excludes PyTorch. The validation command adds the version used by the official image. The evaluator supplies PyTorch, so `submission/requirements.txt` adds no packages.

With Docker and access to the organizers' image, run the template's sandbox check. On a CPU-only Docker host, append `--no-gpu`.

```bash
uv run python validate.py --docker
```

If the published image is unavailable, build it in a separate checkout of the [official starter template](https://github.com/Reinforce-SST/devforge-doom-template) with `docker build -t devforge-doom .`. Then validate this submission with `uv run python validate.py --docker --image devforge-doom`. A prior CPU sandbox check on Apple silicon with an emulated x86 image measured 5.9 ms p50, 12.5 ms p99, and 1,116 MB peak RAM.

The default dry run uses `maps/two_rooms.wad`. For extra checks, pass one `--map path/to/test.wad:MAP01` per map. The dry run checks compatibility and latency. The organizers rank agents on their unseen suite.

## How the agent works

- **Inference.** `DoomAgent` loads the packaged checkpoint once. It runs on one CPU thread even when the runner requests CUDA. `reset()` clears the specialists' memory, router state, and previous action. `act()` samples one specialist through a learned categorical router, masks consecutive USE presses, and returns nine binary button values every four game ticks.
- **Training.** Our recurrent PPO pipeline uses non-evaluation Freedoom2, OBSIDIAN, and procedural maps. It includes reverse keycard drills, exploration rewards, and combat curricula. Extra 13 freezes R2_08, B08, and L4 student 2 and trains the router with keycard supervision and a KL penalty against the parent router outside keycard contexts.
- **Scorer compatibility.** Inference lives under `submission/` and uses only the supplied observations and packaged weights. The evaluated episode needs no network or repository writes. The organizers use their frozen wrapper and base image.

## Validation and results

The public release was tested from a fresh clone with a real Git LFS download. The validator reported **5.9 MB** including weights. A separate Daytona Linux dry run pinned the agent to one CPU and reported **4.3 ms p99** and **270 MB peak resident RAM**. Timing depends on hardware and map. The scorer limits agent compute **per map**, based on the tick cap.

On the team's 37-map proxy suite, Extra 13 scored **2,080.87** versus **1,993.36** for its parent on 10 matched repeats. It recorded **13/30** ordered Tier 3 unlocks on that panel. Across the three confirmation panels (20 repeats total), it gained four ordered Tier 3 unlocks and one Tier 4 exit while giving up four Tier 2 exits. The exact released package scored **1,849.57** versus **1,804.29** in a separate five-repeat check. [Panel details and tradeoffs](brief.pdf)

## Repository layout

| Path | Purpose |
| --- | --- |
| `submission/` | Self-contained inference code and trained weights |
| `training/` | Learner, reward shaping, map generation, and selected-run settings ([guide](training/README.md), [dependencies](training/requirements.txt)) |
| `brief.pdf` | Two-page architecture and results brief |
| `env/`, `validate.py` | Official template wrapper and validator |
| `maps/two_rooms.wad` | Public verification map |
| `docs/` | Organizer [contract](docs/CONTRACT.md), [evaluation](docs/EVALUATION.md), [rules](docs/RULES.md), and [submission guide](docs/SUBMISSION.md) |
