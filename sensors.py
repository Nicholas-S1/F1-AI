
import math
import pygame

from track import ROAD_MASK, point_in_mask

SENSOR_ANGLES = [-70, -35, 0, 35, 70]
MAX_DISTANCE = 150
STEP_SIZE = 5

def on_track(point):
    """Check whether a point is on the asphalt."""
    return point_in_mask(ROAD_MASK, point)


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
