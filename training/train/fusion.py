"""Training-only episode allocation and distillation schedules for fusion runs."""
import math


class EpisodeMix:
    """Select one mutually exclusive sampling bucket per training episode."""

    def __init__(self, weights, source_names, balanced=False):
        self.names = tuple(weights)
        self.weights = tuple(float(weights[name]) for name in self.names)
        supported = {"procedural", "obsidian", "freedoom", "keycard", "combat", "tier3", "tier4"}
        if not set(self.names) <= supported or not all(math.isfinite(w) and w > 0 for w in self.weights):
            raise ValueError("Unknown fusion episode bucket or nonpositive weight")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-9):
            raise ValueError("Fusion episode weights must sum to one")
        self.external = {name: source_names.index(name) for name in self.names if name in ("freedoom", "obsidian")}
        self.counts = dict.fromkeys(self.names, 0)
        self.balanced = balanced
        self.remaining = []
        if balanced and len(set(self.weights)) != 1:
            raise ValueError("Balanced blocks need equal domain weights")

    def draw(self, rng):
        if self.balanced:
            if not self.remaining:self.remaining=list(rng.permutation(len(self.names)))
            name=self.names[self.remaining.pop()]
        else:
            name=self.names[rng.choice(len(self.names), p=self.weights)]
        self.counts[name] += 1
        return name


def guided_kl_weight(start, progress, decay_fraction=1.0):
    if not 0 < decay_fraction <= 1:
        raise ValueError("KL decay fraction must be in (0,1]")
    return start * min(1.0, max(0.0, (1.0 - progress) / decay_fraction))
