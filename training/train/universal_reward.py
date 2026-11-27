"""Training rewards and critic features from the official engine state."""

import math
import struct
from collections import deque
from functools import lru_cache

import gymnasium as gym
import numpy as np
from vizdoom import GameVariable
from doom_env.mapinfo import DOOM_NAMESPACES, parse_udmf, udmf_namespace
from doom_env.sensors import classify, ego_rotate
from doom_env.spaces import ENTITY_TYPES, NUM_RAYS, RAY_HIT_TYPES
from doom_env.wad import map_lumps, read_wad
from eval.progression import sector_path

CRITIC_SIZE = 18
STUCK_WINDOW = 16
STUCK_DISTANCE = 24
DOOM_EXITS = {11, 51, 52, 124, 197, 198}
ZDOOM_EXITS = {243, 244}
LOCKED_RAY = RAY_HIT_TYPES.index("locked_door")
FRONT_RAYS = np.arange(-8, 9) % NUM_RAYS
ENEMY_TYPES = tuple(index for index, name in enumerate(ENTITY_TYPES) if name.startswith("enemy"))


@lru_cache(maxsize=64)
def exit_positions(wad, map_name, geometry):
    """Get candidate exit linedef midpoints when the map format exposes them."""
    if geometry is None:
        return ()
    lumps = map_lumps(read_wad(wad), map_name)
    if "TEXTMAP" in lumps:
        text = lumps["TEXTMAP"].decode("utf-8", "replace")
        specials = [line.get("special", 0) for line in parse_udmf(text).get("linedef", [])]
        kinds = DOOM_EXITS if udmf_namespace(text) in DOOM_NAMESPACES else ZDOOM_EXITS
    else:
        hexen = "BEHAVIOR" in lumps
        fmt = "<HHHB5BHH" if hexen else "<HHHHHHH"
        specials = [entry[3] for entry in struct.iter_unpack(fmt, lumps.get("LINEDEFS", b""))]
        kinds = ZDOOM_EXITS if hexen else DOOM_EXITS
    if len(specials) != len(geometry):
        return ()
    return tuple(center for center, special in zip(geometry, specials) if special in kinds)


class UniversalReward(gym.Wrapper):
    """Return legal observations and keep engine-only features in the info dict."""

    def __init__(self, env, reward_settings=None, *, reverse_stage=None, auxiliary=False,
                 reward_mode=None, movement_penalty=False, freedoom_random_start=False):
        super().__init__(env)
        if reward_mode not in (None, "v2_generalist"):
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        self.freedoom_random_start = freedoom_random_start
        self.reward_mode = reward_mode
        self.movement_penalty = movement_penalty
        self.reverse_stage = reverse_stage
        self.auxiliary = auxiliary
        geometry = env.mapinfo.lines
        midpoints = None if geometry is None else tuple(map(tuple, (geometry.p1 + geometry.p2) * 0.5))
        self.exits = exit_positions(env.wad, env.map_name, midpoints)
        self.doors = tuple(env.mapinfo.door_sectors)
        self.locked = {index for index, lock in env.mapinfo.door_sectors.items() if lock > 0}
        self.reward_settings = {
            "door_open": 0.40, "locked_open": 0.35, "key_pickup": 1.0, "kill": 0.35,
            "exit": 5.0, "damage": 0.01, "door_cross": 0.0, "key_progress": 0.0,
            "exploration": 0.05,
            "damage_dealt": 0.0, "opening_kill": 0.0, "combat_strafe": 0.0, "death": 0.5,
            "lock_opening_progress": 0.0, "lock_reverse_penalty": 0.0,
            "locked_progress": 0.0, "exit_progress": 0.0,
            "backtrack_cross": 0.0, "backtrack_progress": 0.0,
        }
        if self.reward_mode == "v2_generalist":
            self.reward_settings.update(door_open=1.0, locked_open=4.0, key_pickup=4.0,
                                        kill=1.0, exit=15.0, damage=0.02,
                                        exploration=0.02, death=5.0)
        for key, value in (reward_settings or {}).items():
            if key not in self.reward_settings or not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
                raise ValueError(f"Invalid training reward setting {key}={value}")
            self.reward_settings[key] = value
        self.door_positions = {}
        if geometry is not None and (self.reward_settings["locked_progress"] or self.reward_settings["backtrack_progress"]):
            for door in self.doors:
                lines = (geometry.front == door) | (geometry.back == door)
                if lines.any():
                    self.door_positions[door] = tuple(((geometry.p1[lines] + geometry.p2[lines]) * 0.5).mean(0))

    def _objects(self):
        objects = self.env._game.get_state()
        if objects is None:
            return (), 0, None
        keys = []
        enemies = 0
        sector = None
        for obj in objects.objects:
            if obj.category == "Monster":
                enemies += 1
            elif obj.category == "Self":
                sector = obj.sector_id
            elif classify(obj.name, obj.category) == "key":
                keys.append((obj.position_x, obj.position_y))
        return keys, enemies, sector

    def _crossing_reward(self, sector):
        previous = self.player_sector
        self.player_sector = sector
        if sector is None or previous is None or sector == previous:
            return 0.0
        entry = self.door_entry.pop(previous, None)
        if sector in self.doors:
            self.door_entry[sector] = previous
        if entry is None or entry == sector:
            return 0.0
        self.last_crossing = previous, entry, sector
        reward = 0.0
        if self.backtrack_target == (previous, sector) and previous not in self.backtrack_rewarded:
            self.backtrack_rewarded.add(previous)
            self.backtrack_target = None
            reward += self.reward_settings["backtrack_cross"]
        if previous in self.crossed_doors:
            return reward
        self.crossed_doors.add(previous)
        return reward + self.reward_settings["door_cross"]

    def _progress_reward(self, state, keys):
        phase, targets = "exit_progress", self.exits
        if keys:
            phase, targets = "key_progress", keys
        elif self.branch_pending:
            phase, targets = "backtrack_progress", self.branch_target[1]
        elif self.backtrack_target is not None and self.backtrack_target[0] in self.door_positions:
            phase, targets = "backtrack_progress", [self.door_positions[self.backtrack_target[0]]]
        elif (self.eligible_doors & self.locked) - self.opened:
            phase = "locked_progress"
            targets = [point for door, point in self.door_positions.items()
                       if door in self.locked and door in self.eligible_doors and door not in self.opened]
        gain = self.reward_settings[phase]
        if not targets or not gain:
            self.goal_progress = None
            return 0.0
        distance = min(math.hypot(state["x"] - x, state["y"] - y) for x, y in targets)
        signature = phase, tuple(targets)
        if self.goal_progress is None or self.goal_progress[0] != signature:
            self.goal_progress = signature, distance
            return 0.0
        best = self.goal_progress[1]
        self.goal_progress = signature, min(distance, best)
        return gain * min(48.0, max(0.0, best - distance)) / 512

    def _open_doors(self, state):
        ceiling, floor = state["sector_ceiling"], state["sector_floor"]
        return {index for index in self.doors if index < len(ceiling) and ceiling[index] - floor[index] >= 56}

    @staticmethod
    def _cell(state):
        return math.floor(state["x"] / 64), math.floor(state["y"] / 64)

    @staticmethod
    def _relative_nearest(positions, state):
        if not positions:
            return (0.0, 0.0, 0.0, 0.0)
        x, y = state["x"], state["y"]
        goal_x, goal_y = min(positions, key=lambda goal: (goal[0] - x) ** 2 + (goal[1] - y) ** 2)
        forward, right = ego_rotate(goal_x - x, goal_y - y, math.radians(state["angle"]))
        return (np.clip(forward / 2048, -1, 1), np.clip(right / 2048, -1, 1),
                min(1.0, math.hypot(forward, right) / 2048), 1.0)

    def _info(self, info, obs, state, keys, enemies, base_reward, yaw=0.0):
        speed = math.hypot(*obs["vitals"][14:16])
        features = np.asarray(
            (min(len(self.cells) / 128, 1.0), min(speed / 32, 1.0),
             min(self.stuck_counter / 32, 1.0), min(enemies / 64, 1.0), *state["keys"],
             *self._relative_nearest(keys, state), *self._relative_nearest(self.exits, state)),
            dtype=np.float32,
        )
        result = {
            **info,
            "privileged": features,
            "base_reward": base_reward,
            "cells_visited": len(self.cells),
            "doors_opened": len(self.opened),
            "doors_total": len(self.eligible_doors),
            "locked_opened": len(self.opened & self.locked),
            "locked_total": len(self.eligible_doors & self.locked),
            "keys_picked": int(np.count_nonzero(state["keys"])),
            "keys_total": self.keys_total,
            "kills": state["kills"],
            "damage_taken": state["damage_taken"],
            "stuck_steps": self.stuck_steps,
            "dead": state["dead"],
        }
        if self.auxiliary:
            rays = obs["rays"]
            seen = FRONT_RAYS[rays["hit_type"][FRONT_RAYS] == LOCKED_RAY]
            forward = right = 0.0
            if len(seen):
                ray = seen[rays["distance"][seen].argmin()]
                angle = 2 * math.pi * ray / NUM_RAYS
                distance = float(rays["distance"][ray])
                forward, right = distance * math.cos(angle), -distance * math.sin(angle)
            result["auxiliary_targets"] = np.asarray((yaw, bool(len(seen)), forward, right), dtype=np.float32)
        return result

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.freedoom_random_start and self.np_random.random() < 0.5:
            # Map authors place pickups on accessible floors. Sample sectors uniformly.
            game = self.env._game
            anchors = {}
            for obj in game.get_state().objects:
                if obj.name in ("Clip", "Shell", "Stimpack", "Medikit", "HealthBonus", "ArmorBonus"):
                    anchors.setdefault(obj.sector_id, []).append((int(obj.position_x), int(obj.position_y)))
            if anchors:
                points = list(anchors.values())[self.np_random.integers(len(anchors))]
                x, y = points[self.np_random.integers(len(points))]
                game.send_game_command(f"warp {x} {y}")
                game.advance_action(1)
                state = game.get_state()
                if state is None:
                    obs, info = self.env.reset(**kwargs)
                else:
                    obs, info = self.env._observe(state, first=True), self.env._info()
        self.decisions = 0
        if self.reverse_stage in ("lock", "micro_lock"):
            approach = np.zeros(self.action_space.shape, dtype=np.int8)
            approach[0] = 1
            for _ in range(3):
                if obs["vitals"][8:14].any():
                    break
                obs, _, terminated, truncated, info = self.env.step(approach)
                if terminated or truncated:
                    break
            if not obs["vitals"][8:14].any() or self.env.privileged_state()["finished"]:
                raise RuntimeError("Reverse lock start failed to grant the real engine key")
        state = self.previous = self.env.privileged_state()
        if self.reward_settings["damage_dealt"]:
            self.damage_dealt = self.env._game.get_game_variable(GameVariable.DAMAGECOUNT)
        keys, enemies, sector = self._objects()
        self.keys_total = len(keys)
        self.last_objects = keys, enemies, sector
        self.cells = {self._cell(state)}
        if self.reward_mode == "v2_generalist":
            self.novelty_cells = self.cells.copy()
            self.novelty_sectors = set() if sector is None else {sector}
            self.novelty_keys = {int(slot) % 3 for slot in np.flatnonzero(state["keys"])}
        self.eligible_doors = set(self.doors) - self._open_doors(state)
        self.opened = set()
        self.best_clearance = {}
        self.crossed_doors = set()
        self.door_entry = {}
        self.player_sector = sector
        self.last_crossing = self.backtrack_target = None
        self.branch_target = None
        self.branch_pending = False
        if self.reverse_stage == "branch":
            geometry = self.env.mapinfo.lines
            neighbors = {sid: set() for sid in range(self.env.mapinfo.num_sectors)}
            for a, b in zip(geometry.front, geometry.back):
                if b >= 0 and a != b:
                    neighbors[a].add(b)
                    neighbors[b].add(a)
            route = sector_path(neighbors, sector, self.locked)
            main_route = set(sector_path(neighbors, 0, self.locked))
            for a, b in zip(route, route[1:]):
                if b in main_route:
                    portal = ((geometry.front == a) & (geometry.back == b)) | ((geometry.front == b) & (geometry.back == a))
                    centers = (geometry.p1[portal] + geometry.p2[portal]) * 0.5
                    self.branch_target = b, tuple(map(tuple, centers))
                    break
        self.backtrack_rewarded = set()
        self.goal_progress = None
        self._progress_reward(state, keys)
        self.positions = deque([(state["x"], state["y"])], maxlen=STUCK_WINDOW + 1)
        self.stuck_steps = self.stuck_counter = 0
        return obs, self._info(info, obs, state, keys, enemies, 0.0)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.decisions += 1
        state = self.env.privileged_state()
        truncated |= self.reverse_stage == "micro_lock" and self.decisions >= 15 and not terminated
        yaw = math.radians((state["angle"] - self.previous["angle"] + 180) % 360 - 180)
        cell = self._cell(state)
        generalist = self.reward_mode == "v2_generalist"
        novelty = self.novelty_cells if generalist else self.cells
        reward = (-0.001 if generalist else -0.002) + self.reward_settings["exploration"] * (cell not in novelty)
        novelty.add(cell)
        self.cells.add(cell)
        if generalist:
            new_colors = {int(slot) % 3 for slot in np.flatnonzero(state["keys"])} - self.novelty_keys
            if new_colors:
                self.novelty_keys.update(new_colors)
                self.novelty_sectors.clear()
                self.novelty_cells.clear()
                self.novelty_cells.add(cell)

        new_doors = (self._open_doors(state) & self.eligible_doors) - self.opened
        self.opened.update(new_doors)
        settings = self.reward_settings
        reward += settings["door_open"] * len(new_doors) + settings["locked_open"] * len(new_doors & self.locked)
        if settings["lock_opening_progress"] or settings["lock_reverse_penalty"]:
            for door in (self.eligible_doors & self.locked) - (self.opened - new_doors):
                if door >= len(state["sector_ceiling"]):
                    continue
                clearance = min(56, state["sector_ceiling"][door] - state["sector_floor"][door])
                previous = min(56, self.previous["sector_ceiling"][door] - self.previous["sector_floor"][door])
                best = self.best_clearance.get(door, 0.0)
                reward += settings["lock_opening_progress"] * max(0, clearance - best) / 56
                reward -= settings["lock_reverse_penalty"] * max(0, previous - clearance) / 56
                self.best_clearance[door] = max(best, clearance)
        picked = int(state["keys"].sum()) - int(self.previous["keys"].sum())
        reward += settings["key_pickup"] * picked
        kills = state["kills"] - self.previous["kills"]
        reward += settings["kill"] * kills
        if self.decisions <= 96:
            reward += settings["opening_kill"] * kills * (1 - self.decisions / 96)
        if settings["damage_dealt"]:
            dealt = self.env._game.get_game_variable(GameVariable.DAMAGECOUNT)
            reward += settings["damage_dealt"] * max(0, dealt - self.damage_dealt)
            self.damage_dealt = dealt
        if settings["combat_strafe"] and np.isin(obs["visible_entities"]["entity_type"], ENEMY_TYPES).any():
            reward += settings["combat_strafe"] * min(abs(obs["vitals"][15]) / 8, 1)
        reward += settings["exit"] * state["exit_reached"]
        reward -= settings["damage"] * max(0, state["damage_taken"] - self.previous["damage_taken"])
        if state["dead"]:
            reward -= settings["death"]
            if self.reward_mode != "v2_generalist":
                reward -= 0.75 * max(0.0, 1.0 - state["tic"] / state["episode_ticks"])

        self.positions.append((state["x"], state["y"]))
        first, last = self.positions[0], self.positions[-1]
        stuck = len(self.positions) == STUCK_WINDOW + 1 and math.dist(first, last) < STUCK_DISTANCE
        self.stuck_steps += int(stuck)
        self.stuck_counter = (self.stuck_counter + 1) if stuck else 0
        if generalist:
            if self.movement_penalty:
                reward -= 0.02 * stuck
                reward -= 0.005 * (math.dist(first, last) < STUCK_DISTANCE and abs(yaw) > 0.02)
        else:
            reward -= 0.006 * stuck
        self.previous = state
        if not (terminated or truncated):
            self.last_objects = self._objects()
            keys, _, sector = self.last_objects
            if generalist and sector is not None and sector not in self.novelty_sectors:
                reward += 1.0
                self.novelty_sectors.add(sector)
            if settings["door_cross"] or settings["backtrack_cross"] or settings["backtrack_progress"]:
                reward += self._crossing_reward(sector)
            if picked > 0 and self.branch_target:
                self.branch_pending = True
            if self.branch_pending and sector == self.branch_target[0]:
                reward += settings["backtrack_cross"]
                self.branch_pending = False
            if picked > 0 and not self.branch_target and self.last_crossing is not None and (settings["backtrack_cross"] or
                                                                  settings["backtrack_progress"]):
                door, origin, _ = self.last_crossing
                if door not in self.locked and door not in self.backtrack_rewarded:
                    self.backtrack_target = door, origin
            reward += self._progress_reward(state, keys)
        keys, enemies, _ = self.last_objects
        return obs, float(reward), terminated, truncated, self._info(info, obs, state, keys, enemies, base_reward, yaw)
