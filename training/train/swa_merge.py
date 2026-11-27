"""Average 2–4 v1 checkpoints fine-tuned from a common initialization."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from train.train_burst import atomic_save


def merge(paths):
    states = []
    if not 2 <= len(paths) <= 4 or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Select 2–4 distinct checkpoints from the same fine-tuning basin")
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("best_model") or payload.get("model", payload)
        if "ray_type.weight" not in state:
            raise ValueError(f"{path} is not a v1 checkpoint")
        states.append(state)
    reference = states[0]
    if any(state.keys() != reference.keys() for state in states[1:]):
        raise ValueError("Checkpoint parameter names differ")
    merged = {}
    for key, initial in reference.items():
        tensors = [state[key] for state in states]
        if any(tensor.shape != initial.shape or tensor.dtype != initial.dtype
               or not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError(f"Incompatible or nonfinite tensor {key}")
        if initial.is_floating_point():
            merged[key] = torch.stack([tensor.double() for tensor in tensors]).mean(0).to(initial.dtype)
        elif all(torch.equal(initial, tensor) for tensor in tensors):
            merged[key] = initial
        else:
            raise ValueError(f"Nonfloating buffers disagree for {key}")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    args = parser.parse_args()
    weights = merge(args.checkpoints)
    output = ROOT / "submission" / "v1_weights.pt"
    backup = output.with_name(".v1_weights.swa-backup")
    if output.exists():
        os.link(output, backup)
    try:
        atomic_save(output, weights)
        subprocess.run([str(ROOT / ".venv/bin/python"), "validate.py"], cwd=ROOT, check=True)
    except BaseException:
        if backup.exists():
            os.replace(backup, output)
        else:
            output.unlink(missing_ok=True)
        raise
    else:
        backup.unlink(missing_ok=True)
        print(f"Saved and validated {output}")


if __name__ == "__main__":
    main()
