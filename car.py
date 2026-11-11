
import math
import random
import pygame

from track import (
    SMOOTH_TRACK,
    TRACK_WIDTH,
    CHECKPOINTS,
    CHECKPOINT_RADIUS,
    SAFE_MASK,
    point_in_mask,
)


class Car:
    def __init__(self):
        self.attempts = 0
        self.crashes = 0
        self.completed_laps = 0
        self.best_lap = None
        self.last_lap = None

        self.reset()

    def reset(self):
        start = pygame.Vector2(SMOOTH_TRACK[0])
        next_point = pygame.Vector2(SMOOTH_TRACK[1])

        direction = next_point - start

        self.position = start.copy()
        self.angle = math.atan2(direction.y, direction.x)

        self.speed = 0.0
        self.steering = 0.0
        self.throttle = 0.0

        self.lap_time = 0.0
        self.next_checkpoint = 1
        self.attempts += 1

    
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
        # Save previous position for finish detection
        previous_position = self.position.copy()

        # Update attempt timer
        self.lap_time += dt

        
        # Vehicle physics
        acceleration = 160
        braking = 240
        rolling_resistance = 35
        max_speed = 220

        if self.throttle == 1:
            self.speed += acceleration * dt

        elif self.throttle == -1:
            self.speed -= braking * dt

        else:
            self.speed -= rolling_resistance * dt

        # Keep speed between zero and max speed
        self.speed = max(
            0.0,
            min(self.speed, max_speed)
        )

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
        if self.next_checkpoint < len(CHECKPOINTS):
            checkpoint = pygame.Vector2(
                CHECKPOINTS[self.next_checkpoint]
            )

            distance = self.position.distance_to(
                checkpoint
            )

            if distance <= CHECKPOINT_RADIUS:
                self.next_checkpoint += 1
                return "checkpoint"

        # Finish detection after all checkpoints
        else:
            if self.crossed_finish(previous_position):
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
        if self.lap_time >= 60:
            self.reset()
            return "timeout"

        return None

    def crossed_finish(self, previous_position):
        start = pygame.Vector2(SMOOTH_TRACK[0])
        next_point = pygame.Vector2(SMOOTH_TRACK[1])

        forward = (next_point - start).normalize()

        # Calculate how far each position is
        # along the forward direction
        previous_side = (
            previous_position - start
        ).dot(forward)

        current_side = (
            self.position - start
        ).dot(forward)

        # Position across the road
        lateral = (
            self.position - start
        ).cross(forward)

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
        return not point_in_mask(SAFE_MASK, self.position)

    def draw(self, screen):
        forward = pygame.Vector2(
            math.cos(self.angle),
            math.sin(self.angle)
        )

        right = pygame.Vector2(
            -forward.y, forward.x
        )

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

        pygame.draw.polygon(
            screen, (255, 220, 40), vertices
        )

        pygame.draw.circle(
            screen,
            (25, 35, 45),
            (round(p.x), round(p.y)),
            4
        )
