
import math
import pygame

TRACK_POINTS = [
    (280, 655),
    (160, 610),
    (140, 490),
    (200, 400),
    (315, 350),
    (315, 250),
    (260, 175),
    (340, 105),
    (470, 115),
    (590, 170),
    (790, 135),
    (980, 155),
    (1040, 250),
    (980, 340),
    (830, 340),
    (745, 425),
    (815, 515),
    (960, 470),
    (1040, 565),
    (980, 665),
    (795, 690),
    (610, 675),
    (455, 635),
]

TRACK_WIDTH = 78
CURB_WIDTH = 16

ASPHALT = (43, 54, 64)
WHITE = (245, 245, 245)
RED = (230, 45, 50)


def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t

    x = 0.5 * (
        2 * p1[0]
        + (-p0[0] + p2[0]) * t
        + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
        + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
    )

    y = 0.5 * (
        2 * p1[1]
        + (-p0[1] + p2[1]) * t
        + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
        + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
    )

    return (x, y)


def generate_smooth_track():
    smooth = []
    count = len(TRACK_POINTS)

    for i in range(count):
        p0 = TRACK_POINTS[(i - 1) % count]
        p1 = TRACK_POINTS[i]
        p2 = TRACK_POINTS[(i + 1) % count]
        p3 = TRACK_POINTS[(i + 2) % count]

        for step in range(20):
            t = step / 20
            smooth.append(
                catmull_rom(p0, p1, p2, p3, t)
            )

    return smooth


SMOOTH_TRACK = generate_smooth_track()


# CHECKPOINT SETTINGS


CHECKPOINT_COUNT = 12
# Covers the full drivable width (the car crashes
# once it is more than TRACK_WIDTH/2 - 9 px off-center),
# so a car can't drive past a checkpoint without hitting it
CHECKPOINT_RADIUS = 39

CHECKPOINTS = [
    SMOOTH_TRACK[
        i * len(SMOOTH_TRACK) // CHECKPOINT_COUNT
    ]
    for i in range(CHECKPOINT_COUNT)
]


# TRACK DRAWING HELPERS

def draw_smooth_line(screen, points, color, width):
    pygame.draw.lines(
        screen, color, True, points, width
    )

    for x, y in points:
        pygame.draw.circle(
            screen,
            color,
            (round(x), round(y)),
            width // 2
        )


def draw_finish_line(screen):
    x, y = SMOOTH_TRACK[0]
    nx, ny = SMOOTH_TRACK[1]

    angle = math.atan2(ny - y, nx - x)

    dx = -math.sin(angle)
    dy = math.cos(angle)

    forward_x = math.cos(angle)
    forward_y = math.sin(angle)

    squares = 10
    rows = 2

    square_width = TRACK_WIDTH / squares
    square_depth = 7

    for row in range(rows):
        for col in range(squares):

            if (row + col) % 2 == 0:
                color = WHITE
            else:
                color = (15, 15, 15)

            a = -TRACK_WIDTH / 2 + col * square_width
            b = a + square_width

            corners = []

            for side, depth in [
                (a, row * square_depth),
                (b, row * square_depth),
                (b, (row + 1) * square_depth),
                (a, (row + 1) * square_depth),
            ]:
                corners.append((
                    x + dx * side + forward_x * depth,
                    y + dy * side + forward_y * depth
                ))

            pygame.draw.polygon(
                screen, color, corners
            )


# DRAW CHECKPOINTS

_label_font = None


def draw_checkpoints(screen):
    global _label_font

    # Creating a SysFont is slow, so build it once
    if _label_font is None:
        _label_font = pygame.font.SysFont("Arial", 15)

    for i, position in enumerate(CHECKPOINTS):

        # Skip checkpoint 0 because it is
        # our starting/finishing line
        if i == 0:
            continue

        x = round(position[0])
        y = round(position[1])

        pygame.draw.circle(
            screen,
            (50, 220, 100),
            (x, y),
            CHECKPOINT_RADIUS,
            2
        )

        # Label each checkpoint
        label = _label_font.render(
            str(i), True, (255, 255, 255)
        )

        label_rect = label.get_rect(
            center=(x, y)
        )

        screen.blit(label, label_rect)




def draw_track(screen):

    points = SMOOTH_TRACK

    # Draw the white curb base
    draw_smooth_line(
        screen,
        points,
        WHITE,
        TRACK_WIDTH + CURB_WIDTH * 2
    )

    # Draw alternating red curb segments
    distance = 0
    stripe_length = 16

    for i in range(len(points)):

        p1 = pygame.Vector2(points[i])
        p2 = pygame.Vector2(
            points[(i + 1) % len(points)]
        )

        segment = p2 - p1
        length = segment.length()

        if length == 0:
            continue

        direction = segment / length
        position = 0

        while position < length:

            remaining_stripe = (
                stripe_length
                - (distance % stripe_length)
            )

            step = min(
                length - position,
                remaining_stripe
            )

            stripe_index = int(
                (distance + 0.00001)
                / stripe_length
            )

            if stripe_index % 2 == 0:

                start = p1 + direction * position
                end = p1 + direction * (position + step)

                pygame.draw.line(
                    screen,
                    RED,
                    start,
                    end,
                    TRACK_WIDTH + CURB_WIDTH * 2
                )

            distance += step
            position += step

    # Draw thin white inner border
    draw_smooth_line(
        screen,
        points,
        WHITE,
        TRACK_WIDTH + 4
    )

    # Draw asphalt
    draw_smooth_line(
        screen,
        points,
        ASPHALT,
        TRACK_WIDTH
    )

    # Draw checkered start/finish line
    draw_finish_line(screen)

    # Draw all checkpoints
    draw_checkpoints(screen)


# ==========================================
# PRECOMPUTED ROAD MASKS (fast lookups)
# ==========================================
# Checking "is this point on the road?" against every
# track segment is slow. Instead we paint the road once
# into a pixel mask and look points up in O(1).

MASK_SIZE = (1200, 800)


def _build_road_mask(radius):
    """Mask of every pixel within `radius` of the centerline."""
    surface = pygame.Surface(MASK_SIZE)
    surface.fill((0, 0, 0))
    color = (255, 255, 255)
    count = len(SMOOTH_TRACK)

    for i in range(count):
        a = pygame.Vector2(SMOOTH_TRACK[i])
        b = pygame.Vector2(SMOOTH_TRACK[(i + 1) % count])

        # Round cap at each point
        pygame.draw.circle(surface, color, (round(a.x), round(a.y)), radius)

        segment = b - a
        if segment.length_squared() == 0:
            continue

        # Rectangle covering the segment
        normal = pygame.Vector2(-segment.y, segment.x).normalize() * radius
        pygame.draw.polygon(
            surface, color, [a + normal, b + normal, b - normal, a - normal]
        )

    return pygame.mask.from_threshold(
        surface, color, (1, 1, 1, 255)
    )


CAR_RADIUS = 9

# Whole asphalt surface (used by the sensors)
ROAD_MASK = _build_road_mask(TRACK_WIDTH // 2)

# Where the car's center can be without crashing
SAFE_MASK = _build_road_mask(TRACK_WIDTH // 2 - CAR_RADIUS)


def point_in_mask(mask, point):
    x = int(point[0])
    y = int(point[1])

    if 0 <= x < MASK_SIZE[0] and 0 <= y < MASK_SIZE[1]:
        return bool(mask.get_at((x, y)))

    return False
