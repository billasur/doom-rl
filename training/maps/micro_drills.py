"""Short training-only lock exercises with real UDMF key pickups and specials."""

import random

from maps.gen_wads import CARDS, block


def lock_drill(seed):
    rng = random.Random(seed)
    width = rng.choice((48, 56, 64))
    key, lock = rng.choice(CARDS)
    spawn = -rng.randint(70, 83)
    # The training reset moves forward just far enough to collect this physical key.
    things = [
        block("thing", x=spawn, y=0, angle=rng.randrange(-8, 9) % 360, type=1,
              skill1=True, skill2=True, skill3=True, skill4=True, skill5=True, single=True),
        block("thing", x=spawn + 39, y=0, type=key,
              skill1=True, skill2=True, skill3=True, skill4=True, skill5=True, single=True),
    ]
    sectors = [block("sector", texturefloor="FLOOR0_1", textureceiling="CEIL1_1",
                     heightfloor=0, heightceiling=height, lightlevel=192) for height in (128, 0, 128)]
    vertices, sides, lines = {}, [], []

    def line(a, b, front, back=None, **special):
        start, stop = [vertices.setdefault(point, len(vertices)) for point in (a, b)]
        side = len(sides)
        sides.append(block("sidedef", sector=front, texturemiddle="STARTAN2" if back is None else "-",
                           texturetop="BIGDOOR2", texturebottom="-"))
        if back is not None:
            sides.append(block("sidedef", sector=back, texturemiddle="-", texturetop="BIGDOOR2", texturebottom="-"))
            special.update(sideback=side + 1, twosided=True)
        else:
            special["blocking"] = True
        lines.append(block("linedef", v1=start, v2=stop, sidefront=side, **special))

    bounds = (-192, 0, 32, 144)
    for room, (left, right) in enumerate(zip(bounds, bounds[1:])):
        line((left, width), (right, width), room)
        line((right, -width), (left, -width), room)
    line((bounds[0], -width), (bounds[0], width), 0)
    line((bounds[-1], width), (bounds[-1], -width), 2)
    line((0, width), (0, -width), 0, 1, special=13, arg0=0, arg1=32, arg2=150,
         arg3=lock, playeruse=True, repeatspecial=True)
    line((32, width), (32, -width), 1, 2, special=243, playercross=True)
    points = [block("vertex", x=x, y=y) for x, y in vertices]
    return 'namespace = "zdoom";\n\n' + "\n".join(points + sectors + sides + lines + things)
