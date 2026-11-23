
import math
import time
from collections import deque

import pygame

from track import TRACKS, MASK_SIZE
from car import Car, DEFAULT_MAX_SPEED
from sensors import draw_sensors, get_sensor_readings
from agent import QLearningAgent, get_state
from progress import ProgressTracker
from memory import save_memory, load_memory, reset_memory


# ==========================================
# INITIALIZE PYGAME
# ==========================================

pygame.init()

TRACK_AREA_WIDTH, HEIGHT = MASK_SIZE
SIDEBAR_WIDTH = 300
WIDTH = TRACK_AREA_WIDTH + SIDEBAR_WIDTH

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 AI Learning Simulator")

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16, bold=True)
small_font = pygame.font.SysFont("Arial", 14, bold=True)
big_font = pygame.font.SysFont("Arial", 22, bold=True)

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
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.05


# ==========================================
# RACE SETTINGS
# ==========================================

# One entry per car: (color, learning rate, top speed).
# Every value can be changed while running
# (select a car, then use the arrow keys).
CAR_SETTINGS = [
    ((255, 220, 40), 0.05, DEFAULT_MAX_SPEED),   # yellow
    ((235, 60, 60), 0.10, DEFAULT_MAX_SPEED),    # red
    ((60, 140, 255), 0.15, DEFAULT_MAX_SPEED),   # blue
    ((70, 220, 110), 0.20, DEFAULT_MAX_SPEED),   # green
    ((240, 140, 40), 0.25, DEFAULT_MAX_SPEED),   # orange
    ((200, 90, 240), 0.30, DEFAULT_MAX_SPEED),   # purple
]

LEARNING_RATE_STEP = 0.01
MIN_LEARNING_RATE = 0.01
MAX_LEARNING_RATE = 1.0

TOP_SPEED_STEP = 10
MIN_TOP_SPEED = 80
MAX_TOP_SPEED = 400

# SPLIT-TIME REWARD
# When a car reaches a checkpoint (or the finish), its
# time is compared with its own reference time for that
# checkpoint: faster earns a bonus, slower a penalty.
#
# The reference is an average of the car's earlier runs.
# SPLIT_MEMORY controls how much the newest run counts:
#   1.0 = compare with the last run only
#   0.3 = smoothed average of recent runs (less noisy)
SPLIT_MEMORY = 0.3

# Reward per second gained/lost against the reference,
# capped so a single split can't outweigh the +25
# checkpoint reward or the -40 crash penalty
SPLIT_REWARD_PER_SECOND = 5.0
SPLIT_REWARD_CAP = 10.0


# ==========================================
# ONE RACER = CAR + ITS OWN Q-LEARNING AGENT
# ==========================================

class Racer:
    def __init__(self, number, track, color, learning_rate, max_speed):
        self.number = number

        self.car = Car(track, max_speed=max_speed, color=color)

        self.agent = QLearningAgent(
            learning_rate=learning_rate,
            discount_factor=0.95,
            epsilon=1.0
        )

        self.progress = ProgressTracker(track, self.car.position)

        self.learning_updates = 0
        self.last_reward = 0.0

        self.reset_race_stats()
        self.restart_attempt()

    def reset_race_stats(self):
        """Stats that belong to one track."""
        self.best_progress = 0.0
        self.recent_progress = deque(maxlen=100)
        self.average_progress = 0.0

        # Reference time for each checkpoint index
        # (len(checkpoints) = the finish line)
        self.split_reference = {}
        self.last_split_delta = None

        self.eval_attempts = 0
        self.eval_laps = 0
        self.eval_progress = deque(maxlen=100)

    def restart_attempt(self):
        self.car.reset()
        self.progress.reset(self.car.position)
        self.clear_attempt()

    def clear_attempt(self):
        self.current_progress = 0.0
        self.rewarded_distance = 0.0
        self.cancel_decision()

    def cancel_decision(self):
        self.decision_state = None
        self.decision_timer = 0.0
        self.accumulated_reward = 0.0
        self.current_action = "None"

    def set_track(self, track):
        self.car.set_track(track)
        self.progress.reset(self.car.position, track)
        self.reset_race_stats()
        self.clear_attempt()


# ==========================================
# SIMULATION VARIABLES
# ==========================================

track_index = 0
track = TRACKS[track_index]

racers = [
    Racer(i + 1, track, color, learning_rate, max_speed)
    for i, (color, learning_rate, max_speed) in enumerate(CAR_SETTINGS)
]

selected = 0

running = True
paused = False
speed_index = 0
steps_done_last_frame = 0
confirm_reset = False
status_message = ""

# Evaluation mode: the agents always take their best
# known action (no random exploration) and do not
# learn, so we can measure what they have really learned.
eval_mode = False


# ==========================================
# OBSERVE AI STATE
# ==========================================

def observe_car(racer):
    car = racer.car
    track = car.track

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

    # Locate the checkpoint to aim for
    checkpoint_count = len(track.checkpoints)

    if car.next_checkpoint < checkpoint_count:
        options = track.checkpoint_options[car.next_checkpoint]

        # On the shortcut, aim for the checkpoint where
        # it rejoins instead of the skipped ones
        if racer.progress.on_shortcut():
            target_index = options[-1]
        else:
            target_index = options[0]
    else:
        target_index = 0

    target = pygame.Vector2(track.checkpoints[target_index])

    direction = target - car.position

    target_angle = math.atan2(direction.y, direction.x)

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
# SPLIT-TIME REWARD
# ==========================================

def split_reward(racer, checkpoint_index, split_time):
    """Bonus/penalty for reaching a checkpoint faster or
    slower than this car's own earlier runs."""
    reference = racer.split_reference.get(checkpoint_index)

    if reference is None:
        bonus = 0.0
        racer.last_split_delta = None
    else:
        gained = reference - split_time
        racer.last_split_delta = -gained

        bonus = max(
            -SPLIT_REWARD_CAP,
            min(SPLIT_REWARD_CAP, gained * SPLIT_REWARD_PER_SECOND)
        )

    # Evaluation runs don't change the reference times
    if not eval_mode:
        if reference is None:
            racer.split_reference[checkpoint_index] = split_time
        else:
            racer.split_reference[checkpoint_index] = (
                reference + SPLIT_MEMORY * (split_time - reference)
            )

    return bonus


# ==========================================
# ONE TRAINING STEP (one car)
# ==========================================

def training_step(racer, dt):
    car = racer.car
    agent = racer.agent
    track = car.track
    checkpoint_count = len(track.checkpoints)

    # STEP 1: Observe and choose action
    if racer.decision_state is None:

        racer.decision_state = observe_car(racer)

        racer.current_action = agent.choose_action(
            racer.decision_state,
            greedy=eval_mode
        )

        car.apply_action(racer.current_action)

        racer.decision_timer = 0.0
        racer.accumulated_reward = 0.0

    # STEP 2: Move the car
    old_position = car.position.copy()

    result = car.update(dt)

    racer.decision_timer += dt

    # STEP 3: Calculate reward
    reward = -0.1 * dt

    if result not in ("crash", "finish", "timeout"):

        movement_distance = (car.position - old_position).length()

        new_progress, racer.current_progress = (
            racer.progress.update(car.position, movement_distance)
        )

        # Limit reward to valid checkpoint region
        # (on a shortcut track: up to the checkpoint
        # after the optional ones)
        if car.next_checkpoint < checkpoint_count:
            allowed_index = track.checkpoint_options[car.next_checkpoint][-1]
            allowed_distance = track.checkpoint_distances[allowed_index]
        else:
            allowed_distance = track.length

        valid_distance = min(racer.progress.farthest, allowed_distance)

        rewardable_progress = max(
            0.0, valid_distance - racer.rewarded_distance
        )

        reward += rewardable_progress * 0.2

        racer.rewarded_distance = max(
            racer.rewarded_distance, valid_distance
        )

        racer.best_progress = max(
            racer.best_progress, racer.current_progress
        )

    if result == "checkpoint":
        reward += 25
        reward += split_reward(
            racer, car.reached_checkpoint, car.lap_time
        )

    elif result == "crash":
        reward -= 40

    elif result == "finish":
        reward += 200

        if car.last_lap is not None:
            reward += 100 / max(car.last_lap, 1)
            reward += split_reward(
                racer, checkpoint_count, car.last_lap
            )

    elif result == "timeout":
        reward -= 15

    racer.accumulated_reward += reward

    # STEP 4: Detect terminal event
    done = result in ("crash", "finish", "timeout")

    # STEP 5: Q-learning update
    # (skipped in evaluation mode so the Q-table
    # is not changed while we measure it)
    if racer.decision_timer >= DECISION_INTERVAL or done:

        if not eval_mode:
            next_state = observe_car(racer)

            agent.learn(
                racer.decision_state,
                racer.current_action,
                racer.accumulated_reward,
                next_state,
                done=done
            )

            racer.learning_updates += 1

        racer.last_reward = racer.accumulated_reward

        racer.decision_state = None
        racer.decision_timer = 0.0
        racer.accumulated_reward = 0.0

    # STEP 6: Record completed attempt
    if done:

        attempt_progress = min(
            100.0,
            racer.progress.farthest / track.length * 100
        )

        if result == "finish":
            attempt_progress = 100.0

        if eval_mode:
            # Keep evaluation results separate
            # from the training statistics
            racer.eval_attempts += 1
            racer.eval_progress.append(attempt_progress)

            if result == "finish":
                racer.eval_laps += 1
        else:
            racer.recent_progress.append(attempt_progress)

            racer.average_progress = (
                sum(racer.recent_progress) /
                len(racer.recent_progress)
            )

        racer.best_progress = max(
            racer.best_progress, attempt_progress
        )

        # Car.update() already reset the car
        racer.progress.reset(car.position)

        racer.current_progress = 0.0
        racer.rewarded_distance = 0.0

        # Gradually reduce random exploration
        # (only while training)
        if not eval_mode:
            agent.epsilon = max(
                MIN_EPSILON,
                agent.epsilon * EPSILON_DECAY
            )


# ==========================================
# HELPERS FOR KEYBOARD / MOUSE
# ==========================================

def adjust_learning_rate(racer, change):
    agent = racer.agent
    agent.learning_rate = round(
        max(MIN_LEARNING_RATE,
            min(MAX_LEARNING_RATE, agent.learning_rate + change)),
        2
    )


def adjust_top_speed(racer, change):
    car = racer.car
    car.max_speed = max(
        MIN_TOP_SPEED, min(MAX_TOP_SPEED, car.max_speed + change)
    )


def change_track(new_index):
    global track_index, track

    track_index = new_index % len(TRACKS)
    track = TRACKS[track_index]

    # Learned driving skills (Q-tables) carry over;
    # lap times and split references are per track
    for racer in racers:
        racer.set_track(track)


def racer_at(position):
    """Car closest to a mouse click, if any."""
    best = None
    best_distance = 30

    for i, racer in enumerate(racers):
        distance = racer.car.position.distance_to(position)
        if distance < best_distance:
            best, best_distance = i, distance

    return best


# Sidebar layout (filled in while drawing, used for clicks)
leaderboard_rows = []


# ==========================================
# DRAWING
# ==========================================

def format_time(value):
    return f"{value:.2f}s" if value is not None else "--"


def draw_text(text, position, color=(255, 255, 255), use_font=None):
    surface = (use_font or font).render(text, True, color)
    screen.blit(surface, position)
    return surface.get_height()


def draw_sidebar():
    leaderboard_rows.clear()

    x0 = TRACK_AREA_WIDTH
    pygame.draw.rect(screen, (15, 25, 35), (x0, 0, SIDEBAR_WIDTH, HEIGHT))

    x = x0 + 14
    y = 10

    title = "EVALUATION (no learning)" if eval_mode else "TRAINING"
    y += draw_text(f"F1 Q-LEARNING {title}", (x, y), use_font=font) + 2
    y += draw_text(
        f"Track: {track.name}  ({track.length / TRACKS[0].length:.2f}x)",
        (x, y), (180, 200, 220)
    ) + 2

    if paused:
        sim_text = f"Sim Speed: x{SPEED_LEVELS[speed_index]}"
    else:
        real = clock.get_fps() * steps_done_last_frame / 60
        sim_text = (
            f"Sim Speed: x{SPEED_LEVELS[speed_index]} ({real:.0f}x real)"
        )
    y += draw_text(sim_text, (x, y), (180, 200, 220)) + 10

    # ---------- Leaderboard ----------
    y += draw_text("BEST LAPS", (x, y), (255, 210, 80)) + 4

    def lap_sort_key(racer):
        best = racer.car.best_lap
        return (best is None, best or 0, -racer.best_progress)

    ordered = sorted(racers, key=lap_sort_key)

    for place, racer in enumerate(ordered, start=1):
        car = racer.car
        row_rect = pygame.Rect(x0 + 4, y - 2, SIDEBAR_WIDTH - 8, 22)
        leaderboard_rows.append((row_rect, racers.index(racer)))

        if racers.index(racer) == selected:
            pygame.draw.rect(screen, (45, 60, 80), row_rect, border_radius=4)

        pygame.draw.rect(screen, car.color, (x, y + 3, 12, 12))

        if car.best_lap is not None:
            result = format_time(car.best_lap)
        else:
            result = f"{racer.best_progress:.0f}%"

        draw_text(f"{place}. Car {racer.number}", (x + 18, y), use_font=small_font)
        draw_text(result, (x + 110, y), use_font=small_font)
        draw_text(f"{car.completed_laps} laps", (x + 185, y),
                  (180, 200, 220), small_font)
        y += 22

    y += 10

    # ---------- Selected car ----------
    racer = racers[selected]
    car = racer.car
    agent = racer.agent

    pygame.draw.rect(screen, car.color, (x, y + 3, 14, 14))
    y += draw_text(f"   CAR {racer.number} (selected)", (x, y), car.color) + 4

    if racer.last_split_delta is None:
        split_text = "--"
    else:
        split_text = f"{racer.last_split_delta:+.2f}s"

    if racer.eval_progress:
        eval_avg = sum(racer.eval_progress) / len(racer.eval_progress)
        eval_line = (
            f"Eval: {racer.eval_laps}/{racer.eval_attempts} laps, "
            f"avg {eval_avg:.1f}%"
        )
    else:
        eval_line = "Eval: no attempts yet"

    lines = [
        f"Learning Rate: {agent.learning_rate:.2f}   (Up/Down)",
        f"Top Speed: {car.max_speed:.0f}   (Left/Right)",
        f"Epsilon: {agent.epsilon:.3f}",
        f"Attempt: {car.attempts}   Crashes: {car.crashes}",
        f"Checkpoint: {car.next_checkpoint - 1}/{len(track.checkpoints) - 1}",
        f"Progress: {racer.current_progress:.1f}%   Best: {racer.best_progress:.1f}%",
        f"Avg Progress (100): {racer.average_progress:.1f}%",
        f"Lap Time: {car.lap_time:.2f}s   Best: {format_time(car.best_lap)}",
        f"Last Split vs Ref: {split_text}",
        f"Speed: {car.speed:.0f}   Action: {racer.current_action}",
        f"Last Reward: {racer.last_reward:.2f}",
        f"States: {len(agent.q_table)}   Updates: {racer.learning_updates}",
        eval_line,
    ]

    for line in lines:
        y += draw_text(line, (x, y), use_font=small_font) + 3

    y += 10

    # ---------- Controls ----------
    controls = [
        "Click car / TAB: Select car",
        "Up/Down: Learn rate  Left/Right: Speed",
        "(hold Shift for bigger steps)",
        "A: Copy selected settings to all",
        "T: Next track   1-6: Sim speed",
        "SPACE: Pause   E: Eval mode",
        "S: Save  L: Load  M: Reset  ESC: Quit",
    ]

    for line in controls:
        y += draw_text(line, (x, y), (150, 170, 190), small_font) + 2

    if status_message:
        draw_text(f"Status: {status_message}", (x, HEIGHT - 26),
                  (255, 210, 80), small_font)


def draw_world():
    track.draw(screen)

    # Sensors only for the selected car to keep it readable
    draw_sensors(screen, racers[selected].car)

    for i, racer in enumerate(racers):
        if i != selected:
            racer.car.draw(screen)

    # Selected car on top, outlined
    racers[selected].car.draw(screen, highlight=True)


# ==========================================
# MAIN SIMULATION LOOP
# ==========================================

while running:

    clock.tick(60)

    # --------------------------------------
    # HANDLE KEYBOARD / MOUSE INPUT
    # --------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            clicked = racer_at(event.pos)

            if clicked is None:
                for rect, index in leaderboard_rows:
                    if rect.collidepoint(event.pos):
                        clicked = index

            if clicked is not None:
                selected = clicked

        elif event.type == pygame.KEYDOWN:

            shift = event.mod & pygame.KMOD_SHIFT
            racer = racers[selected]

            # Handle memory reset confirmation
            if confirm_reset:

                if event.key == pygame.K_y:
                    reset_memory([r.agent for r in racers])

                    for r in racers:
                        # Cancel the action currently in progress
                        r.cancel_decision()

                        # Clear learning statistics
                        r.learning_updates = 0
                        r.last_reward = 0.0

                    confirm_reset = False
                    paused = True
                    status_message = "Memory RESET!"

                elif event.key in (pygame.K_n, pygame.K_ESCAPE):
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
                status_message = f"Speed x{SPEED_LEVELS[speed_index]}"

            # SELECT CAR
            elif event.key == pygame.K_TAB:
                step = -1 if shift else 1
                selected = (selected + step) % len(racers)

            # PER-CAR LEARNING RATE
            elif event.key in (pygame.K_UP, pygame.K_DOWN):
                change = LEARNING_RATE_STEP * (5 if shift else 1)
                if event.key == pygame.K_DOWN:
                    change = -change
                adjust_learning_rate(racer, change)
                status_message = (
                    f"Car {racer.number} learning rate "
                    f"{racer.agent.learning_rate:.2f}"
                )

            # PER-CAR TOP SPEED
            elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                change = TOP_SPEED_STEP * (5 if shift else 1)
                if event.key == pygame.K_LEFT:
                    change = -change
                adjust_top_speed(racer, change)
                status_message = (
                    f"Car {racer.number} top speed {racer.car.max_speed:.0f}"
                )

            # COPY SELECTED CAR'S SETTINGS TO ALL CARS
            elif event.key == pygame.K_a:
                for r in racers:
                    r.agent.learning_rate = racer.agent.learning_rate
                    r.car.max_speed = racer.car.max_speed
                status_message = f"Car {racer.number} settings -> all"

            # NEXT TRACK
            elif event.key == pygame.K_t:
                change_track(track_index + (-1 if shift else 1))
                status_message = f"Track: {track.name}"

            # SAVE MEMORY
            elif event.key == pygame.K_s:
                paused = True

                save_memory(
                    [r.agent for r in racers],
                    [{"max_speed": r.car.max_speed} for r in racers]
                )

                status_message = f"SAVED {len(racers)} cars"
                print(status_message)

            # LOAD MEMORY
            elif event.key == pygame.K_l:
                paused = True

                extras = load_memory([r.agent for r in racers])

                if extras:
                    for r, extra in zip(racers, extras):
                        if "max_speed" in extra:
                            r.car.max_speed = extra["max_speed"]

                        # Cancel the old action so the
                        # next action uses loaded values
                        r.cancel_decision()

                    status_message = f"LOADED {len(racers)} cars"
                else:
                    status_message = "No saved AI memory found"

                print(status_message)

            # TOGGLE EVALUATION MODE
            elif event.key == pygame.K_e:
                eval_mode = not eval_mode

                # Start fresh attempts so the result
                # is not mixed between the two modes
                for r in racers:
                    r.restart_attempt()

                    if eval_mode:
                        r.eval_attempts = 0
                        r.eval_laps = 0
                        r.eval_progress.clear()

                status_message = (
                    "EVAL mode (no learning)" if eval_mode
                    else "TRAINING mode"
                )

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
            for racer in racers:
                training_step(racer, SIM_DT)

            steps_done_last_frame += 1

            if time.perf_counter() > deadline:
                break

    # ======================================
    # DRAW SIMULATOR
    # ======================================

    draw_world()
    draw_sidebar()

    # --------------------------------------
    # PAUSE AND RESET OVERLAY
    # --------------------------------------

    if paused or confirm_reset:

        overlay = pygame.Surface(
            (TRACK_AREA_WIDTH, HEIGHT), pygame.SRCALPHA
        )

        overlay.fill((0, 0, 0, 130))
        screen.blit(overlay, (0, 0))

        if confirm_reset:
            message = "RESET AI MEMORY (all cars)? Y = Yes / N = No"
        else:
            message = "PAUSED - SPACE to Resume"

        pause_text = big_font.render(message, True, (255, 255, 255))

        screen.blit(
            pause_text,
            pause_text.get_rect(
                center=(TRACK_AREA_WIDTH // 2, HEIGHT // 2)
            )
        )

    pygame.display.flip()

pygame.quit()
