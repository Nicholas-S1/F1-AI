
from track import DEFAULT_TRACK

# Kept for older scripts that use the original track
TRACK_LENGTH = DEFAULT_TRACK.length


def get_track_position(position, track=DEFAULT_TRACK, hint=None):
    """
    Return how far along the track centerline
    the point is, in pixels.

    If `hint` (a segment index from a previous call)
    is given, only nearby segments are searched,
    which is much faster.
    """
    return track.locate(position, hint)[0]


class ProgressTracker:
    def __init__(self, track, position):
        self.track = track
        self.reset(position)

    def reset(self, position, track=None):
        if track is not None:
            self.track = track

        self.previous, self.segment = self.track.locate(position)

        # Distance driven along the track since the start
        # (can go negative if the car drives backwards).
        # Summing small steps avoids the 0% / 100% mix-up
        # right at the start/finish line.
        self.travelled = 0.0
        self.farthest = 0.0

    def on_shortcut(self):
        return self.track.on_shortcut(self.segment)

    def update(self, position, movement_distance):
        track = self.track
        length = track.length

        current, segment = track.locate(
            position, hint=self.segment
        )

        delta = current - self.previous

        # Handle crossing the start/finish line
        if delta > length / 2:
            delta -= length
        elif delta < -length / 2:
            delta += length

        # Reject unrealistic jumps between track sections.
        # (On a shortcut one pixel driven counts as more
        # than one pixel of track, hence the scale.)
        # The jump is not counted, but we still follow the
        # car's new position so tracking can't get stuck
        # (e.g. when cutting the corner where a shortcut
        # splits off, the nearest road suddenly changes).
        limit = movement_distance * track.max_distance_scale + 10
        if abs(delta) > limit:
            self.segment = segment
            self.previous = current
            return 0.0, 0.0

        self.segment = segment
        self.previous = current
        self.travelled += delta

        # Record the farthest position reached
        old_farthest = self.farthest

        self.farthest = max(self.farthest, self.travelled)

        new_progress = max(0.0, self.farthest - old_farthest)

        progress_percent = min(
            100.0, self.farthest / length * 100
        )

        return new_progress, progress_percent
