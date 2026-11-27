"""The Freedoom split and manifest-backed OBSIDIAN map selection."""

import json
import struct
from pathlib import Path

import vizdoom
from doom_env.mapinfo import DOOM_MANUAL_DOORS
from doom_env.wad import map_lumps, read_wad

FREEDOOM_TRAIN = tuple(f"MAP{i:02}" for i in (3, 5, *range(7, 16)))
FREEDOOM_EVAL = ("MAP01", "MAP02", "MAP04", "MAP06")
V2_FREEDOOM_EVAL = ("MAP01", "MAP02", "MAP03", "MAP04", "MAP05", "MAP06", "MAP07", "MAP10", "MAP15")
V2_FREEDOOM_TRAIN = tuple(name for i in range(1, 33) if (name := f"MAP{i:02}") not in V2_FREEDOOM_EVAL)
MAX_OBSIDIAN_SECTORS = 2048


def freedoom_wad():
    return Path(vizdoom.__file__).with_name("freedoom2.wad")


def obsidian_maps(folder, split, *, compact_count=None):
    rows = json.loads((folder / "manifest.json").read_text())
    jobs = []
    for row in rows:
        if not 1 <= row.get("sectors", 0) <= MAX_OBSIDIAN_SECTORS:
            raise ValueError(f"OBSIDIAN map has missing or unsupported sector count: {row['wad']}")
        wad = (folder / row["pack" if split == "train" else "wad"]).resolve()
        if not wad.is_relative_to(folder.resolve()) or not wad.is_file():
            raise ValueError(f"Invalid {split} pool WAD: {wad}")
        jobs.append((wad, row["map"]))
    if len(jobs) != len(set(jobs)) or ((len(jobs) != compact_count if compact_count else len(jobs) < 400) if split == "train" else len(jobs) != 12):
        raise ValueError(f"Invalid {split} pool map count: {len(jobs)}")
    return tuple(jobs)


def obsidian_corridors(folder):
    """Use compact OBSIDIAN levels with multiple ordinary, manually operated doors."""
    jobs = []
    active_wad = None
    ordinary = {special for special, lock in DOOM_MANUAL_DOORS.items() if lock == 0}
    for wad, name in obsidian_maps(folder, "train"):
        if wad != active_wad:
            lumps, active_wad = read_wad(wad), wad
        level = map_lumps(lumps, name)
        if len(level["SECTORS"]) // 26 > 750 or "BEHAVIOR" in level:
            continue
        sectors = [entry[-1] for entry in struct.iter_unpack("<hh8s8s8sH", level["SIDEDEFS"])]
        portals = {sectors[entry[6]] for entry in struct.iter_unpack("<7H", level["LINEDEFS"])
                   if entry[3] in ordinary and entry[6] != 65535}
        if len(portals) >= 2:
            jobs.append((wad, name))
    if not jobs:
        raise ValueError(f"No compact ordinary-door maps in {folder}")
    return tuple(jobs)
