
import pygame
from track import SMOOTH_TRACK

# Convert the track into a list of vectors
POINTS = [pygame.Vector2(p) for p in SMOOTH_TRACK]

# Calculate distance along the circuit
DISTANCES = [0.0]

for i in range(len(POINTS)):
    a = POINTS[i]
    b = POINTS[(i + 1) % len(POINTS)]
    DISTANCES.append(DISTANCES[-1] + (b - a).length())

TRACK_LENGTH = DISTANCES[-1]


def get_track_position(position, hint=None, window=15):
    """
    Return how far along the track centerline
    the car currently is, in pixels.

    If `hint` (a segment index from a previous call)
    is given, only nearby segments are searched,
    which is much faster.
    """
    return _locate(position, hint, window)[0]


def _locate(position, hint=None, window=15):
    """Return (track_position, nearest_segment_index)."""
    p = pygame.Vector2(position)

    nearest_distance = float("inf")
    track_position = 0.0
    nearest_index = 0

    count = len(POINTS)

    if hint is None:
        indices = range(count)
    else:
        indices = (
            (hint + offset) % count
            for offset in range(-window, window + 1)
        )

    for i in indices:
        a = POINTS[i]
        b = POINTS[(i + 1) % count]

        segment = b - a
        length_sq = segment.length_squared()

        if length_sq == 0:
            continue

        t = max(
            0.0,
            min(1.0, (p - a).dot(segment) / length_sq)
        )

        closest = a + segment * t
        distance_sq = (p - closest).length_squared()

        if distance_sq < nearest_distance:
            nearest_distance = distance_sq
            nearest_index = i

            track_position = (
                DISTANCES[i] + t * segment.length()
            )

    return track_position, nearest_index


class ProgressTracker:
    def __init__(self, position):
        self.reset(position)

    def reset(self, position):
        self.previous, self.segment = _locate(position)

        # Distance driven along the track since the start
        # (can go negative if the car drives backwards).
        # Summing small steps avoids the 0% / 100% mix-up
        # right at the start/finish line.
        self.travelled = 0.0
        self.farthest = 0.0

    def update(self, position, movement_distance):
        current, self.segment = _locate(
            position, hint=self.segment
        )

        delta = current - self.previous

        # Handle crossing the start/finish line
        if delta > TRACK_LENGTH / 2:
            delta -= TRACK_LENGTH
        elif delta < -TRACK_LENGTH / 2:
            delta += TRACK_LENGTH

        # Reject unrealistic jumps between track sections
        if abs(delta) > movement_distance + 10:
            return 0.0, 0.0

        self.previous = current
        self.travelled += delta

        # Record the farthest position reached
        old_farthest = self.farthest

        self.farthest = max(
            self.farthest,
            self.travelled
        )

        new_progress = max(
            0.0,
            self.farthest - old_farthest
        )

        progress_percent = min(
            100.0,
            self.farthest / TRACK_LENGTH * 100
        )

        return new_progress, progress_percent
