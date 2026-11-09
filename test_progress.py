
from track import SMOOTH_TRACK
from progress import get_track_position, TRACK_LENGTH

start = get_track_position(SMOOTH_TRACK[0])
middle = get_track_position(
    SMOOTH_TRACK[len(SMOOTH_TRACK) // 2]
)

print("Track length:", round(TRACK_LENGTH, 2))
print("Starting position:", round(start, 2))
print("Halfway position:", round(middle, 2))
