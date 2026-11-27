"""Evaluation-only progression telemetry from player and map state."""

from collections import deque
from itertools import takewhile

import numpy as np
from doom_env.sensors import classify
from doom_env.spaces import BUTTONS, ENTITY_TYPES

APPROACH_RANGE = 96.0
USE_RANGE = 64.0
USE = BUTTONS.index("USE")
ATTACK = BUTTONS.index("ATTACK")
ENEMIES = [ENTITY_TYPES.index(kind) for kind in ("enemy_melee", "enemy_hitscan", "enemy_projectile")]
MILESTONES = ("door_approach", "use_attempt", "key_pickup", "branch_return", "lock_approach", "unlock")


def sector_path(neighbors, start, targets):
    queue = deque([(start,)])
    visited = {start}
    while queue:
        path = queue.popleft()
        sector = path[-1]
        if sector in targets:
            return path
        for neighbor in sorted(neighbors.get(sector, set()) - visited):
            visited.add(neighbor)
            queue.append((*path, neighbor))
    return ()


class Progression:
    def __init__(self, env, tier, state, sector):
        self.env = env
        self.events = []
        self.sector = self.spawn_sector = sector
        self.previous_state = state
        self.opened = set()
        self.nearby = set()
        self.portals = {}
        self.branch_junction = None
        self.branch_sectors = set()
        self.return_pending = False
        self.combat = dict(attack_decisions=0, enemy_visible_decisions=0, attacks_with_enemy_visible=0,
                           first_damage_step=None, first_damage_sector=None, left_spawn_step=None,
                           first_room_clear_step=None, first_room_kills=0)
        self.outside_spawn_steps = 0
        self.spawn_decisions = 0
        self.wall_follow_decisions = 0
        geometry = env.env.mapinfo.lines
        if geometry is None:
            return
        neighbors = {sid: set() for sid in range(env.env.mapinfo.num_sectors)}
        for i, (a, b) in enumerate(zip(geometry.front, geometry.back)):
            a, b = int(a), int(b)
            if b < 0 or b == a:
                continue
            neighbors[a].add(b)
            neighbors[b].add(a)
            for side, door in ((a, b), (b, a)):
                if door in env.eligible_doors:
                    self.portals.setdefault(side, []).append((door, geometry.p1[i], geometry.p2[i]))
        if tier in ("tier3_keycard", "tier4_gauntlet"):
            keys = {int(obj.sector_id) for obj in env.env._game.get_state().objects
                    if classify(obj.name, obj.category) == "key"}
            if len(keys) == 1 and env.locked:
                origin = 0 if env.reverse_stage == "branch" else sector
                key_path = sector_path(neighbors, origin, keys)
                lock_path = sector_path(neighbors, origin, env.locked)
                common = list(takewhile(lambda pair: pair[0] == pair[1], zip(key_path, lock_path)))
                if common and len(common) < len(key_path):
                    self.branch_junction = common[-1][0]
                    self.branch_sectors = set(key_path[len(common):])

    def log(self, event, step, state, sector, **details):
        self.events.append(dict(event=event, step=step, tic=state["tic"], x=round(state["x"], 2),
                                y=round(state["y"], 2), sector=sector, **details))

    def nearby_doors(self):
        state = self.previous_state
        point = np.array((state["x"], state["y"]))
        heading = np.deg2rad(state["angle"])
        direction = np.array((np.cos(heading), np.sin(heading)))
        doors = {}
        for door, a, b in self.portals.get(self.sector, ()):
            edge = b - a
            length_sq = edge @ edge
            if not length_sq:
                continue
            delta = a - point
            distance = float(np.linalg.norm(delta + np.clip((-delta @ edge) / length_sq, 0, 1) * edge))
            if distance > APPROACH_RANGE:
                continue
            cross = direction[0] * edge[1] - direction[1] * edge[0]
            in_use_ray = False
            if abs(cross) > 1e-8:
                t = (delta[0] * edge[1] - delta[1] * edge[0]) / cross
                u = (delta[0] * direction[1] - delta[1] * direction[0]) / cross
                in_use_ray = bool(0 <= t <= USE_RANGE and 0 <= u <= 1)
            previous_distance, previous_ray = doors.get(door, (float("inf"), False))
            doors[door] = (min(distance, previous_distance), previous_ray or in_use_ray)
        return doors

    def record(self, step, action, obs, state, sector):
        before = self.previous_state
        self.spawn_decisions += int(self.sector == self.spawn_sector)
        rays = obs["rays"]
        side_wall = any(rays["hit_type"][i] == 1 and rays["distance"][i] <= 48 / 1024 for i in (16, 48))
        self.wall_follow_decisions += int(side_wall and np.hypot(state["x"] - before["x"], state["y"] - before["y"]) >= 4)
        close = self.nearby_doors()
        for door in sorted(close.keys() - self.nearby):
            distance, in_use_ray = close[door]
            event = "lock_approach" if door in self.env.locked else "door_approach"
            self.log(event, step, before, self.sector, door_sector=door, distance=round(distance, 1),
                     in_use_ray=in_use_ray, keys_held=int(before["keys"].sum()))
        self.nearby = close.keys()
        # This is button intent at the start of a four-tic decision. The engine
        # can turn or move before activating a line, and may ignore a held button.
        if action[USE]:
            door = min(close, key=lambda d: (not close[d][1], close[d][0])) if close else None
            distance, in_use_ray = close[door] if door is not None else (None, False)
            self.log("use_attempt", step, before, self.sector, door_sector=door,
                     distance=round(distance, 1) if distance is not None else None,
                     in_use_ray=in_use_ray, keys_held=int(before["keys"].sum()))

        if self.env.opened != self.opened:
            for door in sorted(self.env.opened - self.opened):
                event = "unlock" if door in self.env.locked else "door_open"
                self.log(event, step, state, sector, door_sector=door)
            self.opened = self.env.opened.copy()
        if np.any(state["keys"] > before["keys"]):
            self.log("key_pickup", step, state, sector,
                     key_slots=np.flatnonzero(state["keys"] > before["keys"]).tolist())
            self.return_pending = sector in self.branch_sectors
        if sector != self.sector and sector is not None:
            self.log("sector_transition", step, state, sector, from_sector=self.sector)
            if self.return_pending and sector == self.branch_junction:
                self.log("branch_return", step, state, sector, from_sector=self.sector)
                self.return_pending = False
            if self.combat["left_spawn_step"] is None and sector != self.spawn_sector:
                self.combat["left_spawn_step"] = step
        if state["damage_taken"] > before["damage_taken"]:
            self.log("damage", step, state, sector, health=state["health"],
                     amount=round(state["damage_taken"] - before["damage_taken"], 2))
            if self.combat["first_damage_step"] is None:
                self.combat.update(first_damage_step=step, first_damage_sector=sector)
        if sector is not None and sector != self.spawn_sector:
            self.outside_spawn_steps += 1
        elif not state["exit_reached"]:
            self.outside_spawn_steps = 0
        if (self.combat["first_room_clear_step"] is None and self.outside_spawn_steps
                and (self.outside_spawn_steps >= 32 or state["exit_reached"]) and state["kills"] > 0
                and not state["dead"]):
            self.combat.update(first_room_clear_step=step, first_room_kills=state["kills"])
        visible = bool(np.isin(obs["visible_entities"]["entity_type"], ENEMIES).any())
        self.combat["enemy_visible_decisions"] += visible
        self.combat["attack_decisions"] += int(action[ATTACK])
        self.combat["attacks_with_enemy_visible"] += int(visible and action[ATTACK])
        if state["finished"]:
            outcome = "exit" if state["exit_reached"] else "death" if state["dead"] else "timeout"
            self.log(outcome, step, state, sector, health=state["health"])
        self.previous_state = state
        self.sector = sector
