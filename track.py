
import math
from collections import deque

import pygame


# ==========================================
# TRACK LAYOUTS
# ==========================================
# Each layout is a closed loop of control points that
# gets smoothed with a Catmull-Rom spline. Cars drive
# the loop in the listed order, starting at point 0.
#
# A layout can also have a SHORTCUT: an extra road that
# leaves the main loop at control point `start`, follows
# the `via` points and rejoins at control point `end`.
# Checkpoints on the main-loop section it bypasses are
# optional, so a car may take either route.

CLASSIC_POINTS = [
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

# Longer circuit that uses the whole screen
LONG_POINTS = [
    (330, 725),
    (170, 715),
    (85, 630),
    (95, 515),
    (195, 455),
    (240, 365),
    (160, 290),
    (95, 200),
    (150, 95),
    (290, 75),
    (400, 150),
    (470, 270),
    (590, 290),
    (640, 190),
    (720, 90),
    (870, 70),
    (1000, 95),
    (1100, 180),
    (1110, 305),
    (1010, 375),
    (880, 365),
    (790, 425),
    (830, 525),
    (960, 520),
    (1080, 545),
    (1115, 650),
    (1030, 725),
    (860, 735),
    (700, 690),
    (580, 600),
    (470, 640),
]

# Same length as Classic, with a shortcut across the
# top of the big U-shaped dip
SHORTCUT_POINTS = [
    (600, 715),
    (410, 720),
    (230, 690),
    (130, 590),
    (120, 440),
    (155, 290),
    (240, 165),
    (370, 105),
    (480, 135),   # 8  shortcut leaves here
    (545, 235),
    (560, 365),
    (630, 465),
    (750, 470),
    (815, 375),
    (810, 245),
    (870, 145),   # 15 shortcut rejoins here
    (990, 110),
    (1085, 185),
    (1105, 330),
    (1050, 450),
    (1080, 580),
    (1005, 690),
    (820, 715),
]

SHORTCUT_ROUTE = {
    "start": 8,
    "via": [(565, 85), (640, 150), (720, 150), (790, 85)],
    "end": 15,
}

# Long circuit with a shortcut that skips the
# hairpin section on the right-hand side
LONG_SHORTCUT_POINTS = [
    (560, 730),
    (380, 735),
    (220, 720),
    (105, 640),
    (85, 520),
    (150, 430),
    (270, 415),
    (370, 470),
    (460, 440),
    (470, 340),
    (380, 280),
    (240, 270),
    (120, 215),
    (120, 110),
    (240, 70),
    (400, 85),
    (520, 130),
    (640, 95),    # 17 shortcut leaves here
    (800, 75),
    (980, 80),
    (1100, 130),
    (1090, 230),
    (960, 250),
    (830, 245),
    (750, 310),
    (800, 385),
    (950, 390),
    (1080, 410),
    (1120, 510),
    (1080, 610),  # 29 shortcut rejoins here
    (980, 700),
    (760, 735),
]

LONG_SHORTCUT_ROUTE = {
    "start": 17,
    "via": [(610, 220), (620, 380), (700, 520), (860, 580)],
    "end": 29,
}


TRACK_WIDTH = 78
CURB_WIDTH = 16

ASPHALT = (43, 54, 64)
WHITE = (245, 245, 245)
RED = (230, 45, 50)

SMOOTH_STEPS = 20

# Covers the full drivable width (the car crashes
# once it is more than TRACK_WIDTH/2 - 9 px off-center),
# so a car can't drive past a checkpoint without hitting it
CHECKPOINT_RADIUS = 39

CAR_RADIUS = 9

MASK_SIZE = (1200, 800)

# How many segments either side of the car's last
# known segment are searched when locating it
SEARCH_WINDOW = 15


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


def generate_smooth_track(control_points):
    """Smooth a closed loop of control points."""
    smooth = []
    count = len(control_points)

    for i in range(count):
        p0 = control_points[(i - 1) % count]
        p1 = control_points[i]
        p2 = control_points[(i + 1) % count]
        p3 = control_points[(i + 2) % count]

        for step in range(SMOOTH_STEPS):
            t = step / SMOOTH_STEPS
            smooth.append(
                catmull_rom(p0, p1, p2, p3, t)
            )

    return smooth


def generate_open_path(control_points):
    """Smooth an open path. The first and last control
    points only set the direction at each end."""
    smooth = []

    for i in range(1, len(control_points) - 2):
        p0, p1, p2, p3 = control_points[i - 1:i + 3]

        for step in range(SMOOTH_STEPS):
            smooth.append(
                catmull_rom(p0, p1, p2, p3, step / SMOOTH_STEPS)
            )

    smooth.append(control_points[-2])
    return smooth


def _polyline_length(points, closed):
    total = 0.0
    count = len(points) if closed else len(points) - 1

    for i in range(count):
        a = points[i]
        b = points[(i + 1) % len(points)]
        total += math.dist(a, b)

    return total


def _point_at_distance(points, distances, target):
    """Point on a closed polyline `target` px from the start."""
    for i in range(len(points)):
        if distances[i + 1] >= target:
            a = points[i]
            b = points[(i + 1) % len(points)]
            span = distances[i + 1] - distances[i]
            t = 0.0 if span == 0 else (target - distances[i]) / span
            return (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
            )

    return points[0]


# ==========================================
# TRACK CLASS
# ==========================================

class Track:
    def __init__(self, name, control_points, shortcut=None):
        self.name = name

        # ---- Main loop ----
        self.main = generate_smooth_track(control_points)
        main_count = len(self.main)

        main_distances = [0.0]
        for i in range(main_count):
            a = self.main[i]
            b = self.main[(i + 1) % main_count]
            main_distances.append(
                main_distances[-1] + math.dist(a, b)
            )

        self.length = main_distances[-1]

        # ---- Shortcut ----
        # Its points are mapped onto the main-loop distances
        # it bypasses, so a car on the shortcut "progresses"
        # faster than one on the main road.
        self.shortcut = []
        self.shortcut_range = None

        if shortcut is not None:
            a_index = shortcut["start"] * SMOOTH_STEPS
            b_index = shortcut["end"] * SMOOTH_STEPS

            assert a_index < b_index, "shortcut can't cross the start line"

            controls = (
                [self.main[a_index - 5], self.main[a_index]]
                + list(shortcut["via"])
                + [self.main[b_index], self.main[(b_index + 5) % main_count]]
            )

            self.shortcut = generate_open_path(controls)
            self.shortcut_range = (
                main_distances[a_index],
                main_distances[b_index],
            )
            self.shortcut_length = _polyline_length(
                self.shortcut, closed=False
            )

        # ---- Segments used for progress tracking ----
        # Each entry: (ax, ay, bx, by, start_distance,
        #              length, track_distance_per_pixel)
        self.segments = []

        for i in range(main_count):
            a = self.main[i]
            b = self.main[(i + 1) % main_count]
            self.segments.append((
                a[0], a[1], b[0], b[1],
                main_distances[i],
                main_distances[i + 1] - main_distances[i],
                1.0,
            ))

        self.main_segment_count = main_count
        self.max_distance_scale = 1.0

        neighbours = [
            [(i - 1) % main_count, (i + 1) % main_count]
            for i in range(main_count)
        ]

        if self.shortcut:
            d_start, d_end = self.shortcut_range
            scale = (d_end - d_start) / self.shortcut_length
            self.max_distance_scale = max(1.0, scale)

            travelled = 0.0
            first = len(self.segments)

            for j in range(len(self.shortcut) - 1):
                a = self.shortcut[j]
                b = self.shortcut[j + 1]
                seg_len = math.dist(a, b)

                self.segments.append((
                    a[0], a[1], b[0], b[1],
                    d_start + travelled * scale,
                    seg_len,
                    scale,
                ))
                travelled += seg_len

                index = len(self.segments) - 1
                neighbours.append([index - 1, index + 1])

            last = len(self.segments) - 1
            neighbours[first][0] = a_index - 1
            neighbours[last][1] = b_index

            # Connect the junctions both ways
            neighbours[a_index - 1].append(first)
            neighbours[a_index].append(first)
            neighbours[b_index - 1].append(last)
            neighbours[b_index].append(last)

        # For each segment, every segment within
        # SEARCH_WINDOW steps along the road network
        self.nearby = [
            self._segments_near(i, neighbours)
            for i in range(len(self.segments))
        ]

        # ---- Checkpoints (evenly spaced by distance) ----
        count = max(8, round(self.length / CHECKPOINT_SPACING))

        self.checkpoint_distances = [
            self.length * i / count for i in range(count)
        ]

        self.checkpoints = [
            _point_at_distance(self.main, main_distances, d)
            for d in self.checkpoint_distances
        ]

        # Checkpoints the shortcut bypasses
        self.skippable = set()

        if self.shortcut_range:
            d_start, d_end = self.shortcut_range
            self.skippable = {
                i for i, d in enumerate(self.checkpoint_distances)
                if d_start < d < d_end
            }
            assert self.skippable, f"{name}: shortcut skips no checkpoint"
            assert max(self.skippable) < count - 1, (
                f"{name}: need a checkpoint after the shortcut"
            )

        # For every "next checkpoint" index, the checkpoints
        # a car may hit next: the next one, plus any further
        # ones reachable by skipping optional checkpoints
        self.checkpoint_options = []

        for next_index in range(count):
            options = [next_index]
            k = next_index
            while k in self.skippable and k + 1 < count:
                k += 1
                options.append(k)
            self.checkpoint_options.append(options)

        # Attempts time out after this many seconds
        # (scaled so longer tracks get more time)
        self.lap_timeout = 60 * self.length / CLASSIC_LENGTH

        # ---- Masks for crash detection / sensors ----
        self.road_mask = self._build_mask(TRACK_WIDTH // 2)
        self.safe_mask = self._build_mask(TRACK_WIDTH // 2 - CAR_RADIUS)

        # Pre-rendered picture of the track
        self._surface = None

    # ------------------------------------------
    # Geometry helpers
    # ------------------------------------------

    def _segments_near(self, start, neighbours):
        seen = {start}
        frontier = deque([(start, 0)])

        while frontier:
            index, depth = frontier.popleft()
            if depth == SEARCH_WINDOW:
                continue
            for n in neighbours[index]:
                if n not in seen:
                    seen.add(n)
                    frontier.append((n, depth + 1))

        return tuple(seen)

    def polylines(self):
        """(points, closed) for every road on the track."""
        lines = [(self.main, True)]
        if self.shortcut:
            lines.append((self.shortcut, False))
        return lines

    def start_pose(self):
        """Start position and heading angle."""
        x, y = self.main[0]
        nx, ny = self.main[1]
        return (x, y), math.atan2(ny - y, nx - x)

    def locate(self, position, hint=None):
        """
        Return (track_distance, segment_index) for the
        nearest point on the road network.

        track_distance is measured along the main loop
        from the start line; shortcut points map onto the
        main-loop stretch they bypass.
        """
        px = position[0]
        py = position[1]

        best = float("inf")
        best_distance = 0.0
        best_index = 0

        if hint is None:
            indices = range(len(self.segments))
        else:
            indices = self.nearby[hint]

        segments = self.segments

        for i in indices:
            ax, ay, bx, by, start, seg_len, scale = segments[i]

            sx = bx - ax
            sy = by - ay
            length_sq = sx * sx + sy * sy

            if length_sq == 0:
                continue

            t = ((px - ax) * sx + (py - ay) * sy) / length_sq
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0

            dx = ax + sx * t - px
            dy = ay + sy * t - py
            d2 = dx * dx + dy * dy

            if d2 < best:
                best = d2
                best_index = i
                best_distance = start + t * seg_len * scale

        return best_distance, best_index

    def on_shortcut(self, segment_index):
        return segment_index >= self.main_segment_count

    # ------------------------------------------
    # Masks
    # ------------------------------------------

    def _build_mask(self, radius):
        """Mask of every pixel within `radius` of a road centerline."""
        surface = pygame.Surface(MASK_SIZE)
        surface.fill((0, 0, 0))
        color = (255, 255, 255)

        for points, closed in self.polylines():
            count = len(points)
            last = count if closed else count - 1

            for i in range(count):
                a = pygame.Vector2(points[i])
                pygame.draw.circle(
                    surface, color, (round(a.x), round(a.y)), radius
                )

                if i >= last:
                    continue

                b = pygame.Vector2(points[(i + 1) % count])
                segment = b - a
                if segment.length_squared() == 0:
                    continue

                normal = pygame.Vector2(
                    -segment.y, segment.x
                ).normalize() * radius

                pygame.draw.polygon(
                    surface, color,
                    [a + normal, b + normal, b - normal, a - normal]
                )

        return pygame.mask.from_threshold(
            surface, color, (1, 1, 1, 255)
        )

    # ------------------------------------------
    # Drawing
    # ------------------------------------------

    def draw(self, screen):
        if self._surface is None:
            self._surface = pygame.Surface(MASK_SIZE)
            self._surface.fill((125, 148, 164))
            self._render(self._surface)

        screen.blit(self._surface, (0, 0))

    def _render(self, screen):
        lines = self.polylines()

        # Each layer is drawn for every road before the
        # next layer, so the asphalt joins cleanly at the
        # shortcut junctions
        for points, closed in lines:
            _draw_smooth_line(
                screen, points, closed, WHITE,
                TRACK_WIDTH + CURB_WIDTH * 2
            )

        for points, closed in lines:
            _draw_curb_stripes(screen, points, closed)

        for points, closed in lines:
            _draw_smooth_line(
                screen, points, closed, WHITE, TRACK_WIDTH + 4
            )

        for points, closed in lines:
            _draw_smooth_line(
                screen, points, closed, ASPHALT, TRACK_WIDTH
            )

        if self.shortcut:
            _draw_shortcut_arrows(screen, self.shortcut)

        _draw_finish_line(screen, self.main)
        self._draw_checkpoints(screen)

    def _draw_checkpoints(self, screen):
        font = pygame.font.SysFont("Arial", 15)

        for i, position in enumerate(self.checkpoints):

            # Skip checkpoint 0 because it is
            # our starting/finishing line
            if i == 0:
                continue

            x = round(position[0])
            y = round(position[1])

            # Optional (skippable) checkpoints are yellow
            color = (240, 200, 60) if i in self.skippable else (50, 220, 100)

            pygame.draw.circle(
                screen, color, (x, y), CHECKPOINT_RADIUS, 2
            )

            label = font.render(str(i), True, (255, 255, 255))
            screen.blit(label, label.get_rect(center=(x, y)))


# ==========================================
# DRAWING HELPERS
# ==========================================

def _draw_smooth_line(screen, points, closed, color, width):
    pygame.draw.lines(screen, color, closed, points, width)

    for x, y in points:
        pygame.draw.circle(
            screen, color, (round(x), round(y)), width // 2
        )


def _draw_curb_stripes(screen, points, closed):
    distance = 0
    stripe_length = 16
    count = len(points)
    last = count if closed else count - 1

    for i in range(last):

        p1 = pygame.Vector2(points[i])
        p2 = pygame.Vector2(points[(i + 1) % count])

        segment = p2 - p1
        length = segment.length()

        if length == 0:
            continue

        direction = segment / length
        position = 0

        while position < length:

            remaining_stripe = (
                stripe_length - (distance % stripe_length)
            )

            step = min(length - position, remaining_stripe)

            stripe_index = int((distance + 0.00001) / stripe_length)

            if stripe_index % 2 == 0:
                start = p1 + direction * position
                end = p1 + direction * (position + step)

                pygame.draw.line(
                    screen, RED, start, end,
                    TRACK_WIDTH + CURB_WIDTH * 2
                )

            distance += step
            position += step


def _draw_shortcut_arrows(screen, points):
    """Faint chevrons showing the shortcut's direction."""
    color = (85, 98, 110)

    for i in range(10, len(points) - 10, 14):
        a = pygame.Vector2(points[i])
        b = pygame.Vector2(points[i + 1])
        if (b - a).length_squared() == 0:
            continue

        forward = (b - a).normalize()
        right = pygame.Vector2(-forward.y, forward.x)

        tip = a + forward * 8
        pygame.draw.lines(
            screen, color, False,
            [tip - forward * 12 + right * 12, tip,
             tip - forward * 12 - right * 12],
            4
        )


def _draw_finish_line(screen, points):
    x, y = points[0]
    nx, ny = points[1]

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

            pygame.draw.polygon(screen, color, corners)


def point_in_mask(mask, point):
    x = int(point[0])
    y = int(point[1])

    if 0 <= x < MASK_SIZE[0] and 0 <= y < MASK_SIZE[1]:
        return bool(mask.get_at((x, y)))

    return False


# ==========================================
# BUILD ALL TRACKS
# ==========================================

# The Classic track had 12 checkpoints; other tracks
# get the same spacing between checkpoints
CLASSIC_LENGTH = _polyline_length(
    generate_smooth_track(CLASSIC_POINTS), closed=True
)
CHECKPOINT_SPACING = CLASSIC_LENGTH / 12

TRACKS = [
    Track("Classic", CLASSIC_POINTS),
    Track("Long", LONG_POINTS),
    Track("Shortcut", SHORTCUT_POINTS, SHORTCUT_ROUTE),
    Track("Long Shortcut", LONG_SHORTCUT_POINTS, LONG_SHORTCUT_ROUTE),
]

DEFAULT_TRACK = TRACKS[0]

# Kept for older scripts that use the original track
SMOOTH_TRACK = DEFAULT_TRACK.main
CHECKPOINTS = DEFAULT_TRACK.checkpoints
