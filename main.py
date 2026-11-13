
import math
import time
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

# Draw the static track once instead of every frame
background = pygame.Surface((WIDTH, HEIGHT))
background.fill((125, 148, 164))
draw_track(background)

DECISION_INTERVAL = 0.1

# Fixed physics step (seconds of simulated time per step)
SIM_DT = 1 / 60

# Simulation steps per frame, chosen with keys 1-6.
# 1 = real time, 200 = up to 200x faster.
SPEED_LEVELS = [1, 5, 20, 50, 100, 200]

# Max real time to spend simulating per frame
FRAME_BUDGET = 1 / 60 * 0.85

# Exploration (epsilon) shrinks by this factor after
# every attempt, down to MIN_EPSILON
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05


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
speed_index = 0
steps_done_last_frame = 0
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

# Evaluation mode: the agent always takes its best
# known action (no random exploration) and does not
# learn, so we can measure what it has really learned.
eval_mode = False
eval_attempts = 0
eval_laps = 0
eval_progress = deque(maxlen=100)


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
# ONE TRAINING STEP
# ==========================================

def training_step(dt):
    global decision_state, current_action
    global decision_timer, accumulated_reward
    global last_reward, learning_updates
    global current_progress, rewarded_distance
    global best_progress, average_progress
    global eval_attempts, eval_laps


    # STEP 1: Observe and choose action
    if decision_state is None:

        decision_state = observe_car(car)

        current_action = agent.choose_action(
            decision_state,
            greedy=eval_mode
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
    # (skipped in evaluation mode so the Q-table
    # is not changed while we measure it)
    if decision_timer >= DECISION_INTERVAL or done:

        if not eval_mode:
            next_state = observe_car(car)

            agent.learn(
                decision_state,
                current_action,
                accumulated_reward,
                next_state,
                done=done
            )

            learning_updates += 1

        last_reward = accumulated_reward

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

        if eval_mode:
            # Keep evaluation results separate
            # from the training statistics
            eval_attempts += 1
            eval_progress.append(attempt_progress)

            if result == "finish":
                eval_laps += 1
        else:
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
        # (only while training)
        if not eval_mode:
            agent.epsilon = max(
                0.05,
                agent.epsilon * 0.999
            )


# ==========================================
# MAIN SIMULATION LOOP
# ==========================================

while running:

    clock.tick(60)

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

            # SIMULATION SPEED (keys 1-6)
            elif pygame.K_1 <= event.key < pygame.K_1 + len(SPEED_LEVELS):
                speed_index = event.key - pygame.K_1
                status_message = (
                    f"Speed x{SPEED_LEVELS[speed_index]}"
                )

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

            # TOGGLE EVALUATION MODE
            elif event.key == pygame.K_e:
                eval_mode = not eval_mode

                # Start a fresh attempt so the result
                # is not mixed between the two modes
                restart_attempt()

                if eval_mode:
                    eval_attempts = 0
                    eval_laps = 0
                    eval_progress.clear()
                    status_message = "EVAL mode (no learning)"
                else:
                    status_message = "TRAINING mode"

            # REQUEST MEMORY RESET
            elif event.key == pygame.K_m:
                paused = True
                confirm_reset = True
                status_message = "Reset memory? Y / N"


    # --------------------------------------
    # AI TRAINING (possibly many steps per frame)
    # --------------------------------------

    if not paused and not confirm_reset and running:

        steps = SPEED_LEVELS[speed_index]

        # Stop early if the frame is taking too long so
        # the window stays responsive at high speeds
        deadline = time.perf_counter() + FRAME_BUDGET
        steps_done_last_frame = 0

        for _ in range(steps):
            training_step(SIM_DT)
            steps_done_last_frame += 1

            if time.perf_counter() > deadline:
                break

    # ======================================
    # DRAW SIMULATOR
    # ======================================

    # Track never changes, so blit the pre-drawn image
    screen.blit(background, (0, 0))
    draw_sensors(screen, car)
    car.draw(screen)

    best_time = (
        f"{car.best_lap:.2f}s"
        if car.best_lap is not None
        else "--"
    )

    if eval_progress:
        eval_avg = sum(eval_progress) / len(eval_progress)
        eval_line = (
            f"Eval: {eval_laps}/{eval_attempts} laps, "
            f"avg {eval_avg:.1f}%"
        )
    else:
        eval_line = "Eval: no attempts yet"

    labels = [
        "F1 Q-LEARNING EVALUATION"
        if eval_mode else
        "F1 Q-LEARNING TRAINING",
        eval_line,
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
        f"Sim Speed: x{SPEED_LEVELS[speed_index]} "
        f"({clock.get_fps() * steps_done_last_frame / 60:.0f}x real)"
        if not paused else
        f"Sim Speed: x{SPEED_LEVELS[speed_index]}",
        f"Action: {current_action}",
        f"Last Reward: {last_reward:.2f}",
        f"Learning Rate: {agent.learning_rate:.2f}",
        f"Epsilon: {agent.epsilon:.3f}",
        f"Learned States: {len(agent.q_table)}",
        f"Q Updates: {learning_updates}",
        "SPACE: Pause / Resume | 1-6: Sim speed",
        "S: Save | L: Load | M: Reset | E: Eval",
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
