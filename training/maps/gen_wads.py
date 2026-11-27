"""Generate deterministic UDMF training and held-out WADs using the standard library."""

import argparse
import random
import struct
from pathlib import Path

TIERS = ("tier1_combat", "tier2_explore", "tier3_keycard", "tier4_gauntlet")
CELL = 96
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
CARDS = ((13, 1), (5, 2), (6, 3))
MONSTERS = (3004, 3004, 3001, 3002, 9)
SUPPLIES = (2011, 2012, 2007, 2007, 2018)
WALLS = ("STONE2", "GRAY1", "STARTAN2")
FLOORS = ("FLOOR0_1", "FLOOR4_8", "FLAT5_4")


def block(kind, **props):
    def literal(value):
        if isinstance(value, str):
            return f'"{value}"'
        return str(value).lower()

    return kind + "\n{\n" + "".join(f"    {key} = {literal(value)};\n" for key, value in props.items()) + "}\n"


def layout(tier, rng):
    if tier == 0:
        return [(0, 0)], [], 0, 0, None, None
    cardinal_sign = rng.choice((-1, 1))
    if tier == 1:
        extra_count = rng.randint(0, 3)
        if extra_count == 0:
            # Three rooms branch off a small hallway junction instead of forming a chain.
            start, end = rng.sample(range(3), 2)
            return [(0, -1), (-1, 0), (1, 0), (0, 0)], [(3, i) for i in range(3)], start, end, None, 3
        rooms = [(0, 0), (1, 0), (1, 1)]
        edges = [(0, 1), (1, 2)]
        candidates = [((2, 0), 1), ((1, -1), 1), ((-1, 0), 0)]
        branch = candidates.pop(rng.randrange(2))
        extras = [branch, *rng.sample(candidates, extra_count - 1)]
        for room, parent in extras:
            edges.append((parent, len(rooms)))
            rooms.append(room)
        start, end = rng.sample(range(len(rooms)), 2)
        return rooms, edges, start, end, None, None
    rooms = [(0, 0), (1, 0), (1, cardinal_sign), (2, 0), (3, 0)]
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    if tier == 3 or rng.choice((False, True)):
        edges.append((1, len(rooms)))
        rooms.append((1, -cardinal_sign))
    if tier == 3 and rng.choice((False, True)):
        edges.append((2, len(rooms)))
        rooms.append((2, cardinal_sign))
    return rooms, edges, 0, 4, 2, None


def textmap(tier, seed, *, combat_warmup=False, reverse_stage=None, combat_drill=None, monster_density=1.0):
    if (combat_warmup or combat_drill) and tier != 3:
        raise ValueError("Combat drills are defined for Tier 4")
    if reverse_stage is not None and (tier != 2 or reverse_stage not in ("lock", "branch")):
        raise ValueError("Reverse starts are defined for Tier 3")
    rng = random.Random(seed)
    rooms, edges, spawn_room, exit_room, key_room, junction = layout(tier, rng)
    rotation = rng.randrange(4)
    centers = []
    for x, y in rooms:
        for _ in range(rotation):
            x, y = -y, x
        centers.append((6 * x, 6 * y))

    tiles, sectors, doors, things = {}, [], {}, []

    def sector(floor, height=128):
        index = len(sectors)
        sectors.append(block("sector", texturefloor=floor, textureceiling="CEIL1_1",
                             heightfloor=0, heightceiling=height, lightlevel=rng.randint(160, 208)))
        return index

    room_tiles = []
    for index, (x, y) in enumerate(centers):
        if index == junction:
            tiles[x, y] = sector("FLOOR0_1")
            room_tiles.append([])
            continue
        radius_x, radius_y = (
            (rng.randint(2, 4), rng.randint(2, 3)) if tier == 0 else (rng.randint(1, 2), rng.randint(1, 2))
        )
        floor = sector(rng.choice(FLOORS))
        cells = [(x + i, y + j) for i in range(-radius_x, radius_x + 1) for j in range(-radius_y, radius_y + 1)]
        if tier == 0:
            # Keep the center aisle and room perimeter clear around the pillars.
            cover = [(x + i, y + j) for i in range(1 - radius_x, radius_x) if i
                     for j in range(1 - radius_y, radius_y) if j]
            pillars = rng.sample(cover, min(len(cover), rng.randint(1, 3)))
            cells = [cell for cell in cells if cell not in pillars]
        tiles.update(dict.fromkeys(cells, floor))
        room_tiles.append(cells)

    key_type, lock_number = rng.choice(CARDS)
    shelter_door = None
    lock_start = None
    for start, end in edges:
        ax, ay = centers[start]
        bx, by = centers[end]
        dx, dy = (bx - ax) // 6, (by - ay) // 6
        passage = [(ax + dx * i, ay + dy * i) for i in range(1, 6) if (ax + dx * i, ay + dy * i) not in tiles]
        locked = key_room is not None and end == exit_room
        planned_door = locked or tier == 1 or rng.random() < 0.65
        needs_regular_door = tier == 3 and end == len(rooms) - 1 and 0 not in doors.values()
        if (planned_door or needs_regular_door
                or ((combat_warmup or combat_drill) and start == spawn_room)
                or (reverse_stage == "branch" and end == key_room)):
            door = sector("FLAT5_4", height=0)
            door_cell = passage.pop(len(passage) // 2)
            tiles[door_cell] = door
            doors[door] = lock_number if locked else 0
            if locked:
                lock_start = ((door_cell[0] - dx, door_cell[1] - dy), DIRECTIONS.index((dx, dy)) * 90)
            if (combat_warmup or combat_drill) and start == spawn_room:
                shelter_door = door
        if passage:
            tiles.update(dict.fromkeys(passage, sector("FLOOR0_1")))

    used = set()

    def place(kind, cell, angle, offset=(0, 0)):
        x, y = cell
        used.add(cell)
        things.append(block("thing", x=x * CELL + CELL // 2 + offset[0], y=y * CELL + CELL // 2 + offset[1],
                            angle=angle, type=kind, skill1=True, skill2=True, skill3=True,
                            skill4=True, skill5=True, single=True))

    spawn = centers[spawn_room]
    spawn_angle = rng.randrange(8) * 45
    if combat_warmup or combat_drill:
        first_room = next(end for start, end in edges if start == spawn_room)
        dx, dy = (centers[first_room][i] - spawn[i] for i in range(2))
        spawn_angle = DIRECTIONS.index((dx // 6, dy // 6)) * 90
    if reverse_stage == "lock":
        spawn, spawn_angle = lock_start
        key_offset = tuple(39 * component for component in DIRECTIONS[spawn_angle // 90])
        spawn_angle = (spawn_angle + rng.choice((-14, 0, 14))) % 360
    elif reverse_stage == "branch":
        spawn = centers[key_room]
        dx, dy = rng.choice(DIRECTIONS)
        nearby_key = (spawn[0] + dx, spawn[1] + dy)
        spawn_angle = DIRECTIONS.index((dx, dy)) * 90
    place(1, spawn, spawn_angle)
    # A corner tile stays clear of corridor connections and is reachable on two sides.
    exit_cell = rng.choice([cell for cell in room_tiles[exit_room]
                            if cell[0] != centers[exit_room][0] and cell[1] != centers[exit_room][1]])
    exit_sector = sector("GATE3")
    tiles[exit_cell] = exit_sector
    used.add(exit_cell)
    if key_room is not None:
        key_cell = spawn if reverse_stage == "lock" else nearby_key if reverse_stage == "branch" else rng.choice(room_tiles[key_room])
        place(key_type, key_cell, 0, key_offset if reverse_stage == "lock" else (0, 0))

    outside_spawn = [cell for index, cells in enumerate(room_tiles) if index != spawn_room for cell in cells]
    all_cells = [cell for cells in room_tiles for cell in cells]
    enemy_cells = outside_spawn or all_cells
    enemy_count = (rng.randint(1, 5), 0, 0, rng.randint(6, 12))[tier]
    if combat_warmup:
        enemy_count = rng.randint(2, 4)
    if combat_drill:
        enemy_count = rng.randint(4, 7)
        # Introduce one early ambusher just beyond the protected first corridor.
        approach = sorted(room_tiles[first_room], key=lambda cell: abs(cell[0] - spawn[0]) + abs(cell[1] - spawn[1]))
        ambush = next(cell for cell in approach if cell not in used)
        place(rng.choice((3004, 3001)), ambush, (spawn_angle + 180) % 360)
        enemy_count -= 1
    if monster_density != 1.0 and enemy_count:
        enemy_count = max(1, round(enemy_count * monster_density))
    for _ in range(enemy_count):
        place(rng.choice(MONSTERS), rng.choice([cell for cell in enemy_cells if cell not in used]), rng.randrange(8) * 45)
    for _ in range((4, 4, 5, 10)[tier]):
        place(rng.choice(SUPPLIES), rng.choice([cell for cell in all_cells if cell not in used]), 0)

    boundaries = {}
    for (x, y), room in tiles.items():
        corners = ((x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y))
        for a, b in zip(corners, corners[1:] + corners[:1]):
            key = tuple(sorted((a, b)))
            if key in boundaries:
                boundaries[key].append(room)
            else:
                boundaries[key] = [a, b, room]

    vertices, sides, lines = {}, [], []
    wall_texture = rng.choice(WALLS)

    def vertex(point):
        return vertices.setdefault(point, len(vertices))

    def side(room, **props):
        index = len(sides)
        sides.append(block("sidedef", sector=room, **props))
        return index

    for a, b, front, *back_room in boundaries.values():
        back = back_room[0] if back_room else None
        if front == back:
            continue
        flags = {}
        if back is None:
            flags.update(sidefront=side(front, texturemiddle="DOORTRAK" if front in doors else wall_texture),
                         blocking=True)
        else:
            if front in doors or front == exit_sector:
                a, b, front, back = b, a, back, front
            flags.update(twosided=True)
            if back in doors:
                lock = doors[back]
                flags.update(special=13 if lock else 12, arg0=0, arg1=16, arg2=150,
                             playeruse=True, repeatspecial=True)
                if lock:
                    flags.update(arg3=lock)
                elif back != shelter_door:
                    flags.update(monsteruse=True)
            if back == exit_sector:
                flags.update(special=243, playercross=True)
            flags.update(sidefront=side(front, texturetop="BIGDOOR2" if back in doors else "-"),
                         sideback=side(back))
        lines.append(block("linedef", v1=vertex(a), v2=vertex(b), **flags))
    points = [block("vertex", x=x * CELL, y=y * CELL) for x, y in vertices]
    return 'namespace = "zdoom";\n\n' + "\n".join(points + sectors + sides + lines + things)


def write_wad(path, text):
    write_udmf_pack(path, [text])


def write_udmf_pack(path, texts):
    body = bytearray()
    directory = bytearray()
    for index, text in enumerate(texts, 1):
        lumps = ((f"MAP{index:02}".encode(), b""), (b"TEXTMAP", text.encode("ascii")), (b"ENDMAP", b""))
        for name, data in lumps:
            directory.extend(struct.pack("<ii8s", 12 + len(body), len(data), name))
            body.extend(data)
    path.write_bytes(struct.pack("<4sii", b"PWAD", len(directory) // 16, 12 + len(body)) + body + directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--split", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--train-per-tier", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.train_per_tier <= 100:
        parser.error("--train-per-tier must be between 1 and 100 to keep tier seeds disjoint")
    for split, count, base in (("train", args.train_per_tier, 10_000), ("eval", 3, 90_000)):
        if args.split not in ("all", split):
            continue
        for tier, label in enumerate(TIERS):
            destination = args.root / split / label
            destination.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                seed = base + 100 * tier + index
                path = destination / f"seed_{seed}.wad"
                if not path.exists():
                    write_wad(path, textmap(tier, seed))
        print(f"{split}: {count * len(TIERS)} WADs at {args.root / split}")


if __name__ == "__main__":
    main()
