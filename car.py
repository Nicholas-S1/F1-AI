
import math
import pygame

from track import (
    DEFAULT_TRACK,
    TRACK_WIDTH,
    CHECKPOINT_RADIUS,
    point_in_mask,
)

DEFAULT_MAX_SPEED = 220


class Car:
    def __init__(
        self,
        track=DEFAULT_TRACK,
        max_speed=DEFAULT_MAX_SPEED,
        color=(255, 220, 40),
    ):
        self.track = track
        self.max_speed = max_speed
        self.color = color

        self.reset_stats()
        self.reset()

    def reset_stats(self):
        self.attempts = 0
        self.crashes = 0
        self.completed_laps = 0
        self.best_lap = None
        self.last_lap = None

    def set_track(self, track):
        """Move the car to a new track and clear its lap stats."""
        self.track = track
        self.reset_stats()
        self.reset()

    def reset(self):
        start, angle = self.track.start_pose()

        self.position = pygame.Vector2(start)
        self.angle = angle

        self.speed = 0.0
        self.steering = 0.0
        self.throttle = 0.0

        self.lap_time = 0.0
        self.next_checkpoint = 1
        self.attempts += 1

        # Index of the checkpoint just reached
        # (set when update() returns "checkpoint")
        self.reached_checkpoint = None

    def apply_action(self, action):
        # Determine steering direction
        if "left" in action:
            self.steering = -1

        elif "right" in action:
            self.steering = 1

        else:
            self.steering = 0

        # Determine acceleration or braking
        if "accelerate" in action:
            self.throttle = 1

        elif "brake" in action:
            self.throttle = -1

        else:
            self.throttle = 0

    def update(self, dt):
        track = self.track

        # Save previous position for finish detection
        previous_position = self.position.copy()

        # Update attempt timer
        self.lap_time += dt

        # Vehicle physics
        acceleration = 160
        braking = 240
        rolling_resistance = 35

        if self.throttle == 1:
            self.speed += acceleration * dt

        elif self.throttle == -1:
            self.speed -= braking * dt

        else:
            self.speed -= rolling_resistance * dt

        # Keep speed between zero and this car's max speed
        self.speed = max(0.0, min(self.speed, self.max_speed))

        # Steering
        steering_strength = 2.5
        speed_factor = min(self.speed / 75, 1.0)

        self.angle += (
            self.steering
            * steering_strength
            * speed_factor
            * dt
        )

        # Forward movement
        movement = pygame.Vector2(
            math.cos(self.angle),
            math.sin(self.angle)
        ) * self.speed * dt

        self.position += movement

        # Crash detection
        if self.has_crashed():
            self.crashes += 1
            self.reset()
            return "crash"

        # Checkpoint detection
        checkpoint_count = len(track.checkpoints)

        if self.next_checkpoint < checkpoint_count:

            # Usually just the next checkpoint, but on a
            # track with a shortcut the car may skip ahead
            # past the optional checkpoints
            for index in track.checkpoint_options[self.next_checkpoint]:
                checkpoint = track.checkpoints[index]

                if self.position.distance_to(checkpoint) <= CHECKPOINT_RADIUS:
                    self.next_checkpoint = index + 1
                    self.reached_checkpoint = index
                    return "checkpoint"

        # Finish detection after all checkpoints
        elif self.crossed_finish(previous_position):
            self.completed_laps += 1
            self.last_lap = self.lap_time

            if (
                self.best_lap is None
                or self.lap_time < self.best_lap
            ):
                self.best_lap = self.lap_time

            self.reset()
            return "finish"

        # Prevent attempts from lasting forever
        if self.lap_time >= track.lap_timeout:
            self.reset()
            return "timeout"

        return None

    def crossed_finish(self, previous_position):
        start = pygame.Vector2(self.track.main[0])
        next_point = pygame.Vector2(self.track.main[1])

        forward = (next_point - start).normalize()

        # Calculate how far each position is
        # along the forward direction
        previous_side = (previous_position - start).dot(forward)
        current_side = (self.position - start).dot(forward)

        # Position across the road
        lateral = (self.position - start).cross(forward)

        # Must cross the finish in the
        # correct direction, on the road
        return (
            previous_side < 0
            and current_side >= 0
            and abs(lateral) <= TRACK_WIDTH / 2
        )

    def has_crashed(self):
        # Crashed once the car's center leaves the safe
        # zone (road width minus the car's radius)
        return not point_in_mask(self.track.safe_mask, self.position)

    def draw(self, screen, highlight=False):
        forward = pygame.Vector2(
            math.cos(self.angle),
            math.sin(self.angle)
        )

        right = pygame.Vector2(-forward.y, forward.x)

        p = self.position

        vertices = [
            p + forward * 17,
            p + forward * 8 + right * 7,
            p - forward * 12 + right * 7,
            p - forward * 17 + right * 10,
            p - forward * 17 - right * 10,
            p - forward * 12 - right * 7,
            p + forward * 8 - right * 7,
        ]

        pygame.draw.polygon(screen, self.color, vertices)

        if highlight:
            pygame.draw.polygon(screen, (255, 255, 255), vertices, 2)

        pygame.draw.circle(
            screen,
            (25, 35, 45),
            (round(p.x), round(p.y)),
            4
        )
