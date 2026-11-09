
import math
from collections import deque

import pygame

from track import draw_track, CHECKPOINTS
from car import Car
from sensors import draw_sensors, get_sensor_readings
from agent import QLearningAgent, get_state
from progress import (
    ProgressTracker,
    TRACK_LENGTH,
    get_track_position
)
from memory import save_memory, load_memory, reset_memory


# ==========================================
# INITIALIZE PYGAME
# ==========================================

pygame.init()

WIDTH = 1200
HEIGHT = 800

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 AI Learning Simulator")

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20, bold=True)

DECISION_INTERVAL = 0.1


# ==========================================
# CREATE CAR AND Q-LEARNING AGENT
# ==========================================

car = Car()

agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0
)

progress = ProgressTracker(car.position)

checkpoint_distances = [
    get_track_position(point)
    for point in CHECKPOINTS
]


# ==========================================
# SIMULATION VARIABLES
# ==========================================

running = True
paused = False
confirm_reset = False
status_message = ""

current_action = "None"
last_reward = 0.0
learning_updates = 0

best_progress = 0.0
current_progress = 0.0
rewarded_distance = 0.0

decision_timer = 0.0
decision_state = None
accumulated_reward = 0.0

recent_progress = deque(maxlen=100)
average_progress = 0.0


# ==========================================
# RESET CURRENT ATTEMPT
# ==========================================

def restart_attempt():
    global current_progress
    global rewarded_distance
    global decision_state
    global decision_timer
    global accumulated_reward
    global current_action

    car.reset()
    progress.reset(car.position)

    current_progress = 0.0
    rewarded_distance = 0.0

    decision_state = None
    decision_timer = 0.0
    accumulated_reward = 0.0
    current_action = "None"


# ==========================================
# OBSERVE AI STATE
# ==========================================

def observe_car(car):
    readings = get_sensor_readings(car)

    distances = [
        distance for distance, endpoint in readings
    ]

    sensor_state = get_state(distances)

    # Categorize speed
    if car.speed < 60:
        speed_state = "slow"
    elif car.speed < 140:
        speed_state = "medium"
    else:
        speed_state = "fast"

    # Locate next checkpoint
    if car.next_checkpoint < len(CHECKPOINTS):
        target = pygame.Vector2(
            CHECKPOINTS[car.next_checkpoint]
        )
    else:
        target = pygame.Vector2(CHECKPOINTS[0])

    direction = target - car.position

    target_angle = math.atan2(
        direction.y,
        direction.x
    )

    angle_difference = (
        target_angle - car.angle + math.pi
    ) % (2 * math.pi) - math.pi

    if angle_difference < -0.25:
        direction_state = "left"
    elif angle_difference > 0.25:
        direction_state = "right"
    else:
        direction_state = "straight"

    return sensor_state + (
        speed_state,
        direction_state
    )


# ==========================================
# MAIN SIMULATION LOOP
# ==========================================

while running:

    dt = min(clock.tick(60) / 1000.0, 0.05)

    # --------------------------------------
    # HANDLE KEYBOARD INPUT
    # --------------------------------------

    
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            # Handle memory reset confirmation
            if confirm_reset:

                if event.key == pygame.K_y:
                    reset_memory(agent)

                    # Cancel the action currently in progress
                    decision_state = None
                    decision_timer = 0.0
                    accumulated_reward = 0.0
                    current_action = "None"

                    # Clear learning statistics
                    learning_updates = 0
                    last_reward = 0.0

                    confirm_reset = False
                    paused = True
                    status_message = "Memory RESET!"

                    print(
                        "RESET: States =",
                        len(agent.q_table),
                        "Epsilon =",
                        agent.epsilon
                    )

                elif event.key in (
                    pygame.K_n,
                    pygame.K_ESCAPE
                ):
                    confirm_reset = False
                    paused = True
                    status_message = "Reset cancelled"

            elif event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_SPACE:
                paused = not paused

            elif event.key == pygame.K_p:
                paused = True

            elif event.key == pygame.K_r:
                paused = False

            # SAVE MEMORY
            elif event.key == pygame.K_s:
                paused = True

                save_memory(agent)

                status_message = (
                    f"SAVED {len(agent.q_table)} states"
                )

                print(status_message)

            # LOAD MEMORY
            elif event.key == pygame.K_l:
                paused = True

                if load_memory(agent):

                    # Cancel the old action so the
                    # next action uses loaded values
                    decision_state = None
                    decision_timer = 0.0
                    accumulated_reward = 0.0
                    current_action = "None"

                    status_message = (
                        f"LOADED {len(agent.q_table)} states"
                    )

                    print(
                        status_message,
                        "| Epsilon:",
                        agent.epsilon
                    )

                else:
                    status_message = (
                        "No saved AI memory found"
                    )
                    print(status_message)

            # REQUEST MEMORY RESET
            elif event.key == pygame.K_m:
                paused = True
                confirm_reset = True
                status_message = "Reset memory? Y / N"


    # --------------------------------------
    # AI TRAINING
    # --------------------------------------

    if not paused and not confirm_reset and running:

        # STEP 1: Observe and choose action
        if decision_state is None:

            decision_state = observe_car(car)

            current_action = agent.choose_action(
                decision_state
            )

            car.apply_action(current_action)

            decision_timer = 0.0
            accumulated_reward = 0.0

        # STEP 2: Move the car
        old_position = car.position.copy()

        result = car.update(dt)

        decision_timer += dt

        # STEP 3: Calculate reward
        reward = -0.1 * dt

        if result not in (
            "crash",
            "finish",
            "timeout"
        ):

            movement_distance = (
                car.position - old_position
            ).length()

            new_progress, current_progress = (
                progress.update(
                    car.position,
                    movement_distance
                )
            )

            # Limit reward to valid checkpoint region
            if car.next_checkpoint < len(CHECKPOINTS):
                allowed_distance = checkpoint_distances[
                    car.next_checkpoint
                ]
            else:
                allowed_distance = TRACK_LENGTH

            valid_distance = min(
                progress.farthest,
                allowed_distance
            )

            rewardable_progress = max(
                0.0,
                valid_distance - rewarded_distance
            )

            reward += rewardable_progress * 0.2

            rewarded_distance = max(
                rewarded_distance,
                valid_distance
            )

            best_progress = max(
                best_progress,
                current_progress
            )

        if result == "checkpoint":
            reward += 25

        elif result == "crash":
            reward -= 40

        elif result == "finish":
            reward += 200

            if car.last_lap is not None:
                reward += 100 / max(
                    car.last_lap, 1
                )

        elif result == "timeout":
            reward -= 15

        accumulated_reward += reward

        # STEP 4: Detect terminal event
        done = result in (
            "crash",
            "finish",
            "timeout"
        )

        # STEP 5: Q-learning update
        if decision_timer >= DECISION_INTERVAL or done:

            next_state = observe_car(car)

            agent.learn(
                decision_state,
                current_action,
                accumulated_reward,
                next_state,
                done=done
            )

            last_reward = accumulated_reward
            learning_updates += 1

            decision_state = None
            decision_timer = 0.0
            accumulated_reward = 0.0

        # STEP 6: Record completed attempt
        if done:

            attempt_progress = min(
                100.0,
                progress.farthest / TRACK_LENGTH * 100
            )

            if result == "finish":
                attempt_progress = 100.0

            recent_progress.append(attempt_progress)

            average_progress = (
                sum(recent_progress) /
                len(recent_progress)
            )

            best_progress = max(
                best_progress,
                attempt_progress
            )

            # Car.update() already reset the car
            progress.reset(car.position)

            current_progress = 0.0
            rewarded_distance = 0.0

            # Gradually reduce random exploration
            agent.epsilon = max(
                0.05,
                agent.epsilon * 0.999
            )

    # ======================================
    # DRAW SIMULATOR
    # ======================================

    screen.fill((125, 148, 164))

    draw_track(screen)
    draw_sensors(screen, car)
    car.draw(screen)

    best_time = (
        f"{car.best_lap:.2f}s"
        if car.best_lap is not None
        else "--"
    )

    labels = [
        "F1 Q-LEARNING TRAINING",
        f"Attempt: {car.attempts}",
        f"Crashes: {car.crashes}",
        f"Completed Laps: {car.completed_laps}",
        f"Checkpoint: {car.next_checkpoint - 1}/"
        f"{len(CHECKPOINTS) - 1}",
        f"Track Progress: {current_progress:.1f}%",
        f"Best Progress: {best_progress:.1f}%",
        f"Avg Progress (100): {average_progress:.1f}%",
        f"Lap Time: {car.lap_time:.2f}s",
        f"Best Lap: {best_time}",
        f"Speed: {car.speed:.0f}",
        f"Action: {current_action}",
        f"Last Reward: {last_reward:.2f}",
        f"Learning Rate: {agent.learning_rate:.2f}",
        f"Epsilon: {agent.epsilon:.3f}",
        f"Learned States: {len(agent.q_table)}",
        f"Q Updates: {learning_updates}",
        "SPACE: Pause / Resume",
        "S: Save | L: Load | M: Reset",
        "ESC: Quit",
        f"Status: {status_message}",
    ]

    panel = pygame.Surface(
        (390, len(labels) * 25 + 18),
        pygame.SRCALPHA
    )

    panel.fill((15, 25, 35, 205))
    screen.blit(panel, (10, 10))

    for i, label in enumerate(labels):

        text = font.render(
            label,
            True,
            (255, 255, 255)
        )

        screen.blit(
            text,
            (20, 20 + i * 25)
        )

    # --------------------------------------
    # PAUSE AND RESET OVERLAY
    # --------------------------------------

    if paused or confirm_reset:

        overlay = pygame.Surface(
            (WIDTH, HEIGHT),
            pygame.SRCALPHA
        )

        overlay.fill((0, 0, 0, 130))
        screen.blit(overlay, (0, 0))

        if confirm_reset:
            message = "RESET AI MEMORY? Y = Yes / N = No"
        else:
            message = "PAUSED - SPACE to Resume"

        pause_text = font.render(
            message,
            True,
            (255, 255, 255)
        )

        screen.blit(
            pause_text,
            pause_text.get_rect(
                center=(WIDTH // 2, HEIGHT // 2)
            )
        )

    pygame.display.flip()

pygame.quit()
