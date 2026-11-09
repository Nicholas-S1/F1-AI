
import math
import pygame

from track import SMOOTH_TRACK, TRACK_WIDTH

SENSOR_ANGLES = [-70, -35, 0, 35, 70]
MAX_DISTANCE = 150
STEP_SIZE = 5

# Cache track segments once
SEGMENTS = [
    (
        pygame.Vector2(SMOOTH_TRACK[i]),
        pygame.Vector2(
            SMOOTH_TRACK[(i + 1) % len(SMOOTH_TRACK)]
        )
    )
    for i in range(len(SMOOTH_TRACK))
]


def on_track(point):
    """Check whether a point is on the asphalt."""
    p = pygame.Vector2(point)
    radius_squared = (TRACK_WIDTH / 2) ** 2

    for a, b in SEGMENTS:
        segment = b - a
        length_squared = segment.length_squared()

        if length_squared == 0:
            continue

        t = max(
            0.0,
            min(1.0, (p - a).dot(segment) / length_squared)
        )

        nearest = a + segment * t

        if (p - nearest).length_squared() <= radius_squared:
            return True

    return False


def get_sensor_readings(car):
    """Return five distances and their endpoints."""
    readings = []

    for offset in SENSOR_ANGLES:
        angle = car.angle + math.radians(offset)

        direction = pygame.Vector2(
            math.cos(angle),
            math.sin(angle)
        )

        distance = 0

        while distance < MAX_DISTANCE:
            point = car.position + direction * distance

            if not on_track(point):
                break

            distance += STEP_SIZE

        distance = min(distance, MAX_DISTANCE)
        endpoint = car.position + direction * distance

        readings.append((distance, endpoint))

    return readings


def draw_sensors(screen, car):
    readings = get_sensor_readings(car)

    for distance, endpoint in readings:
        start = (
            round(car.position.x),
            round(car.position.y)
        )

        end = (
            round(endpoint.x),
            round(endpoint.y)
        )

        pygame.draw.line(
            screen,
            (60, 255, 100),
            start,
            end,
            2
        )

        pygame.draw.circle(
            screen,
            (255, 70, 70),
            end,
            4
        )

    return readings
