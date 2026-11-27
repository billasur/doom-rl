"""Spatial recurrent actor-critic using the legal telemetry observation."""

from math import cos, pi, sin
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.v0_net import ENTITY_SCALE, VITALS_SCALE

RAYS = 64
ENTITIES = 32
GRID = CHANNELS = 16
CELL_UNITS = 64
TELEMETRY_SIZE = RAYS * 2 + ENTITIES * 7 + 17
HISTORY_SIZE = 18 + RAYS
ACTION_COMBINATIONS = ((), (0,), (1,), (2,), (3,), (4,), (5,), (0, 2), (0, 3),
                       (0, 4), (0, 5), (7,), (0, 7), (6,), (0, 6), (2, 6), (3, 6),
                       (4, 6), (5, 6), (1, 6), (1, 2, 6), (1, 3, 6))
ACTION_TABLE = np.zeros((len(ACTION_COMBINATIONS), 9), dtype=np.int8)
for _index, _buttons in enumerate(ACTION_COMBINATIONS):
    ACTION_TABLE[_index, list(_buttons)] = 1
ACTION_TABLE.setflags(write=False)
ACTION_COUNT = len(ACTION_TABLE)
USE_ACTIONS = (11, 12)
CATEGORICAL_HISTORY_SIZE = ACTION_COUNT + 1
DEPTH_ROWS, DEPTH_COLUMNS = 8, 32
DEPTH_SIZE = DEPTH_ROWS * DEPTH_COLUMNS
OPTIONAL_MODULES = ("yaw_correction", "lock_head", "actor_history", "depth_encoder", "depth_projection", "visit_bias")
_SCAN_SHIFTS = (np.arange(RAYS)[None, :] + np.arange(-8, 9)[:, None]) % RAYS


def policy_width(*, action_history=False, categorical=False, depth=False, visits=False, **_):
    history = CATEGORICAL_HISTORY_SIZE if categorical else HISTORY_SIZE * action_history
    return TELEMETRY_SIZE + history + DEPTH_SIZE * depth + visits


def mask_use(logits, prev_used):
    logits = logits.float().clone()
    logits[..., 11:13] = logits[..., 11:13].masked_fill(prev_used[..., None], -1e9)
    return logits


def pack_telemetry(obs):
    """Pack single or vector observations without copying the image buffers."""
    rays, entities = obs["rays"], obs["visible_entities"]
    batch = obs["vitals"].shape[:-1]
    return np.concatenate(
        (rays["distance"], rays["hit_type"], entities["features"].reshape(*batch, ENTITIES * 5),
         entities["entity_type"], entities["mask"], obs["vitals"]),
        axis=-1,
        dtype=np.float32,
    )


def pack_policy(obs, history=None, depth=False):
    telemetry = pack_telemetry(obs)
    if history is not None:
        telemetry = history.pack(telemetry)
    if depth:
        pixels = np.asarray(obs.get("depth"))
        if pixels.shape == (120, 160):
            strip = pixels.reshape(DEPTH_ROWS, 15, DEPTH_COLUMNS, 5).mean(axis=(1, 3), dtype=np.float32) / 255
        else:
            strip = np.zeros((DEPTH_ROWS, DEPTH_COLUMNS), dtype=np.float32)
        telemetry = np.concatenate((telemetry, strip.ravel()))
    return telemetry


class ActionHistory:
    """One-hot state for each of nine buttons plus the four-step ray-distance change."""

    def __init__(self, *, action_history=True, visits=False, categorical=False):
        self.scans = deque(maxlen=4)
        self.action = None
        self.action_history = action_history
        self.categorical = categorical
        self.visits = visits
        self.heading = 0.0
        self.position = np.zeros(2, dtype=np.float32)
        self.previous_scan = None
        self.visited = deque(maxlen=256)

    def visit_feature(self, telemetry):
        scan = np.where(telemetry[RAYS:2 * RAYS] >= 4, 1, telemetry[:RAYS])
        if self.previous_scan is not None:
            errors = np.minimum(np.abs(self.previous_scan[_SCAN_SHIFTS] - scan), 0.12).mean(axis=1)
            errors += 0.001 * np.abs(np.arange(-8, 9))
            self.heading += (errors.argmin() - 8) * (2 * pi / RAYS)
            forward, right = telemetry[TELEMETRY_SIZE - 3:TELEMETRY_SIZE - 1]
            c, s = cos(self.heading), sin(self.heading)
            self.position += 4 * np.array([c * forward + s * right, s * forward - c * right])
        self.previous_scan = scan
        cell = tuple(np.floor(self.position / CELL_UNITS + 0.5))
        revisit = cell in self.visited
        if not self.visited or self.visited[-1] != cell:
            self.visited.append(cell)
        return np.array([revisit], dtype=np.float32)

    def pack(self, telemetry):
        parts = [telemetry]
        scan = telemetry[:RAYS]
        if self.action_history or self.categorical:
            delta = scan - self.scans[0] if len(self.scans) == 4 else np.zeros(RAYS, np.float32)
            self.scans.append(scan.copy())
            if self.categorical:
                previous = np.zeros(CATEGORICAL_HISTORY_SIZE, np.float32)
                if self.action is not None:
                    previous[self.action] = 1
                previous[-1] = np.abs(delta).mean()
                parts.append(previous)
            else:
                buttons = np.zeros((9, 2), np.float32)
                if self.action is not None:
                    buttons[np.arange(9), self.action] = 1
                parts.extend((buttons.ravel(), delta))
        if self.visits:
            parts.append(self.visit_feature(telemetry))
        return np.concatenate(parts)


def checkpoint_features(weights):
    return {
        "reset_visit_on_key": "key_phase_reset" in weights,
        "pose_aux": "yaw_correction.0.weight" in weights,
        "action_history": "actor_history.0.weight" in weights,
        "categorical": weights["actor.weight"].shape[0] == ACTION_COUNT,
        "depth": "depth_projection.weight" in weights,
        "visits": "visit_bias.weight" in weights,
    }


def load_spatial_weights(model, weights, *, allow_new=False):
    if not allow_new:
        return model.load_state_dict(weights)
    weights = dict(weights)
    if not model.reset_visit_on_key:
        weights.pop("key_phase_reset", None)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any(key.split(".", 1)[0] not in OPTIONAL_MODULES for key in missing if key != "key_phase_reset"):
        raise ValueError(f"Checkpoint does not match spatial policy: missing {missing}, unexpected {unexpected}")


class SpatialState(NamedTuple):
    grid: torch.Tensor
    hidden: torch.Tensor
    scan: torch.Tensor

    def reset_where(self, starts):
        return SpatialState(self.grid.masked_fill(starts[:, None, None, None], 0),
                            self.hidden.masked_fill(starts[:, None], 0),
                            self.scan.masked_fill(starts[:, None], -1))


class SpatialActorCritic(nn.Module):
    hidden_size = 256
    privileged_size = 18

    def __init__(self, observation_space, action_space, *, pose_aux=False, action_history=False,
                 depth=False, visits=False, categorical=False, reset_visit_on_key=False):
        super().__init__()
        self.reset_visit_on_key = reset_visit_on_key
        if reset_visit_on_key:
            self.register_buffer("key_phase_reset", torch.tensor(True))
        if action_space.shape != (9,):
            raise ValueError(f"Expected MultiBinary(9), got {action_space}")
        ray_types = int(observation_space["rays"]["hit_type"].nvec.max())
        entity_types = int(observation_space["visible_entities"]["entity_type"].nvec.max())
        self.ray_type = nn.Embedding(ray_types, 8)
        self.ray_conv = nn.Sequential(nn.Conv1d(9, 16, 5, padding=2, padding_mode="circular"), nn.SiLU(),
                                      nn.Conv1d(16, 16, 3, padding=1, padding_mode="circular"), nn.SiLU())
        self.ray_summary = nn.Sequential(nn.AvgPool1d(4), nn.Flatten(), nn.Linear(256, 64), nn.SiLU())
        self.entity_type = nn.Embedding(entity_types, 8)
        self.entity_features = nn.Sequential(nn.Linear(13, 32), nn.LayerNorm(32), nn.SiLU())
        self.query = nn.Linear(64 + 17, 32)
        self.empty_entity = nn.Parameter(torch.zeros(1, 1, 32))
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.project_rays = nn.Linear(16, CHANNELS)
        self.project_entities = nn.Linear(32, CHANNELS)
        self.grid_update = nn.Conv2d(CHANNELS * 2, CHANNELS, 1)
        self.grid_summary = nn.Sequential(nn.Conv2d(CHANNELS, 16, 3, stride=2, padding=1), nn.SiLU(),
                                          nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(256, 64), nn.SiLU())
        self.memory = nn.GRUCell(64 + 17 + 32 + 64, self.hidden_size)
        self.categorical = categorical
        self.history_size = CATEGORICAL_HISTORY_SIZE if categorical else HISTORY_SIZE * action_history
        actions = ACTION_COUNT if categorical else 9
        self.actor = nn.Linear(self.hidden_size + CATEGORICAL_HISTORY_SIZE * categorical, actions)
        if categorical:
            self.register_buffer("action_table", torch.tensor(ACTION_TABLE), persistent=False)
            stamp = torch.zeros(1, 1, GRID, GRID, dtype=torch.bool)
            stamp[..., GRID // 2, GRID // 2] = True
            self.register_buffer("visited_stamp", stamp, persistent=False)
        self.critic = nn.Sequential(nn.Linear(self.hidden_size + self.privileged_size, 128),
                                    nn.SiLU(), nn.Linear(128, 1))
        self.next_ray = nn.Sequential(nn.Linear(self.hidden_size, RAYS), nn.Sigmoid())
        angles = torch.arange(RAYS, dtype=torch.float32) * (2 * pi / RAYS)
        shifts = torch.arange(-8, 9)
        self.register_buffer("bearings", torch.stack((-angles.sin(), -angles.cos()), dim=-1), persistent=False)
        self.register_buffer("shift_indices", (torch.arange(RAYS)[None] + shifts[:, None]) % RAYS, persistent=False)
        self.register_buffer("yaw_angles", shifts * (2 * pi / RAYS), persistent=False)
        self.register_buffer("vitals_scale", torch.tensor(VITALS_SCALE), persistent=False)
        self.register_buffer("entity_scale", torch.tensor(ENTITY_SCALE), persistent=False)
        self.yaw_correction = self.lock_head = self.actor_history = None
        if pose_aux:
            self.yaw_correction = nn.Sequential(nn.Linear(RAYS * 2, 48), nn.SiLU(), nn.Linear(48, 1))
            self.lock_head = nn.Linear(self.hidden_size, 3)
            nn.init.zeros_(self.yaw_correction[-1].weight)
            nn.init.zeros_(self.yaw_correction[-1].bias)
        if action_history and not categorical:
            self.actor_history = nn.Sequential(nn.Linear(self.hidden_size + HISTORY_SIZE, 64),
                                               nn.SiLU(), nn.Linear(64, 9))
            nn.init.zeros_(self.actor_history[-1].weight)
            nn.init.zeros_(self.actor_history[-1].bias)
        self.depth_encoder = self.depth_projection = self.visit_bias = None
        if depth:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(), nn.Flatten(),
            )
            self.depth_projection = nn.Linear(16 * 2 * 8, 64 + 17, bias=False)
            nn.init.zeros_(self.depth_projection.weight)
        if visits:
            self.visit_bias = nn.Linear(1, actions, bias=False)
            nn.init.zeros_(self.visit_bias.weight)

    def initial_state(self, reference):
        batch = reference.shape[0]
        return SpatialState(reference.new_zeros((batch, CHANNELS, GRID, GRID)),
                            reference.new_zeros((batch, self.hidden_size)),
                            reference.new_full((batch, RAYS + int(self.reset_visit_on_key)), -1))

    def estimate_yaw(self, previous, scan):
        """Align consecutive circular scans because the contract has no yaw channel."""
        previous, scan = previous[:, :RAYS], scan[:, :RAYS]
        difference = (previous[:, self.shift_indices] - scan[:, None]).abs().clamp_max(0.12)
        costs = difference.mean(dim=-1) + 0.01 * self.yaw_angles.abs()
        weights = (-costs / 0.008).softmax(dim=-1)
        estimated = (weights * self.yaw_angles).sum(dim=-1)
        if self.yaw_correction is not None:
            residual = self.yaw_correction(torch.cat((previous.clamp_min(0), scan), dim=-1))
            estimated = estimated + 0.5 * residual.squeeze(-1).float().tanh()
        return estimated.masked_fill(previous[:, 0] < 0, 0.0)

    def warp(self, state, yaw, velocity):
        c, s = yaw.cos(), yaw.sin()
        # Four ticks per decision. affine_grid maps output cells back into the previous grid.
        scale = 8.0 / (GRID * CELL_UNITS)
        x, y = velocity[:, 1] * scale, -velocity[:, 0] * scale
        matrix = torch.stack((c, s, c * x + s * y, -s, c, -s * x + c * y), dim=-1).reshape(-1, 2, 3)
        coordinates = F.affine_grid(matrix, state.grid.shape, align_corners=False)
        return F.grid_sample(state.grid, coordinates, align_corners=False)

    def splat(self, distance, hit, entities, mask, ray_features, entity_features):
        # Contract distances are expressed in units of 1024. Grid x is right and y is back.
        entity_bearings = -entities[..., 1:3]
        rays_xy = distance[..., None] * self.bearings
        entity_xy = entities[..., :1] * entity_bearings
        xy = torch.cat((rays_xy, entity_xy), dim=1) * (1024 / CELL_UNITS) + GRID / 2
        cells = xy.floor().long()
        inside = ((cells >= 0) & (cells < GRID)).all(dim=-1)
        valid = inside & torch.cat((hit != 0, mask), dim=-1)
        indices = (cells[..., 1] * GRID + cells[..., 0]).clamp(0, GRID ** 2 - 1)
        features = torch.cat((self.project_rays(ray_features), self.project_entities(entity_features)), dim=1)
        weighted = features * valid[..., None]
        batch = features.shape[0]
        projected = features.new_zeros((batch, GRID ** 2, CHANNELS))
        projected = projected.scatter_add(1, indices[..., None].expand_as(weighted), weighted)
        counts = features.new_zeros((batch, GRID ** 2)).scatter_add(1, indices, valid.to(features.dtype))
        return (projected / counts.clamp_min(1)[..., None]).transpose(1, 2).reshape(batch, CHANNELS, GRID, GRID)

    def encode(self, telemetry):
        distance, hit, entities, classes, mask, vitals = telemetry[..., :TELEMETRY_SIZE].split(
            (64, 64, 160, 32, 32, 17), dim=-1
        )
        hit, classes, mask = hit.long(), classes.long(), mask.bool()
        entities = entities.reshape(-1, ENTITIES, 5)
        distance, vitals_normalized = distance.clamp(0, 1), (vitals / self.vitals_scale).clamp(-1, 1)
        ray_features = self.ray_conv(torch.cat((distance[..., None], self.ray_type(hit)), dim=-1).transpose(1, 2))
        ray_summary = self.ray_summary(ray_features)
        entity_features = self.entity_features(
            torch.cat(((entities / self.entity_scale).clamp(-1, 1), self.entity_type(classes)), dim=-1))
        fused = torch.cat((ray_summary, vitals_normalized), dim=-1)
        if self.depth_encoder is not None and telemetry.shape[-1] >= TELEMETRY_SIZE + DEPTH_SIZE:
            strip = telemetry[..., -DEPTH_SIZE:].reshape(-1, 1, DEPTH_ROWS, DEPTH_COLUMNS)
            fused = fused + self.depth_projection(self.depth_encoder(strip))
        keys = torch.cat((entity_features, self.empty_entity.expand(telemetry.shape[0], -1, -1)), dim=1)
        # The null slot makes attention well-defined even when every entity slot is empty.
        padding = F.pad(~mask, (0, 1), value=False)
        context, _ = self.attention(self.query(fused)[:, None], keys, keys, key_padding_mask=padding,
                                    need_weights=False)

        scan = torch.where(hit >= 4, torch.ones_like(distance), distance)
        if self.reset_visit_on_key:
            scan = torch.cat((scan, vitals[:, 8:14].sum(-1, keepdim=True)), dim=-1)
        projected = self.splat(distance, hit, entities, mask, ray_features.transpose(1, 2), entity_features)
        return fused, context[:, 0], scan, projected, vitals[:, 14:16]

    def actor_logits(self, hidden, history):
        actor_input = torch.cat((hidden, history[..., :CATEGORICAL_HISTORY_SIZE]), dim=-1) if self.categorical else hidden
        logits = self.actor(actor_input)
        if self.actor_history is not None:
            logits = logits + self.actor_history(torch.cat((hidden, history[..., :HISTORY_SIZE]), dim=-1))
        if self.visit_bias is not None:
            offset = self.history_size
            logits = logits + self.visit_bias(history[..., offset:offset + 1])
        return logits

    def recurrent(self, encoded, state, privileged=None, predict_rays=False, history=None):
        fused, context, scan, projected, velocity = encoded
        grid = self.advance_grid(state, self.estimate_yaw(state.scan, scan), projected, velocity, scan)
        hidden = self.memory(torch.cat((fused, context, self.grid_summary(grid)), dim=-1), state.hidden)
        value = self.critic(torch.cat((hidden, privileged), dim=-1)).squeeze(-1) if privileged is not None else None
        next_ray = self.next_ray(hidden) if predict_rays else None
        return self.actor_logits(hidden, history), value, next_ray, SpatialState(grid, hidden, scan)

    def advance_grid(self, state, yaw, projected, velocity, scan):
        warped = self.warp(state, yaw, velocity)
        if self.reset_visit_on_key:
            gained = (state.scan[:, -1] >= 0) & (scan[:, -1] > state.scan[:, -1])
            warped = torch.cat((warped[:, :-1], warped[:, -1:].masked_fill(gained[:, None, None, None], 0)), dim=1)
        grid = 0.9 * warped + 0.35 * self.grid_update(torch.cat((warped, projected), dim=1)).tanh()
        if self.categorical:
            visited = (0.995 * warped[:, -1:]).masked_fill(self.visited_stamp, 1.0)
            grid = torch.cat((grid[:, :-1], visited), dim=1)
        return grid

    def sequence(self, telemetry, state, starts, privileged, *, predict_pose=False):
        """Batch stateless layers across T×B and scan only the recurrent states."""
        shape = telemetry.shape[:2]
        fused, context, scans, projections, velocities = (
            item.unflatten(0, shape) for item in self.encode(telemetry.flatten(0, 1))
        )
        grids, yaws = [], []
        grid, hidden, previous_scan = state
        for scan, projection, velocity, reset in zip(scans, projections, velocities, starts):
            reset_grid = grid.masked_fill(reset[:, None, None, None], 0)
            reset_scan = previous_scan.masked_fill(reset[:, None], -1)
            yaw = self.estimate_yaw(reset_scan, scan)
            grid = self.advance_grid(SpatialState(reset_grid, hidden, reset_scan), yaw, projection, velocity, scan)
            grids.append(grid)
            if predict_pose:
                yaws.append(yaw)
            previous_scan = scan
        summaries = self.grid_summary(torch.cat(grids, dim=0)).unflatten(0, shape)
        memory_input = torch.cat((fused, context, summaries), dim=-1)
        hiddens = []
        for frame, reset in zip(memory_input, starts):
            hidden = self.memory(frame, hidden.masked_fill(reset[:, None], 0))
            hiddens.append(hidden)
        hidden = torch.stack(hiddens)
        values = self.critic(torch.cat((hidden, privileged), dim=-1)).squeeze(-1)
        outputs = self.actor_logits(hidden, telemetry[..., TELEMETRY_SIZE:]), values, self.next_ray(hidden)
        if predict_pose:
            return (*outputs, torch.stack(yaws), self.lock_head(hidden))
        return outputs

    def forward(self, telemetry, state=None, privileged=None, predict_rays=False):
        state = self.initial_state(telemetry) if state is None else state
        return self.recurrent(self.encode(telemetry), state, privileged, predict_rays,
                              telemetry[..., TELEMETRY_SIZE:])
