
import random


class QLearningAgent:
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Stores learned Q-values
        self.q_table = {}

        # Available driving actions
        self.actions = [
            "accelerate_straight",
            "accelerate_left",
            "accelerate_right",
            "brake_straight",
            "brake_left",
            "brake_right",
            "coast_straight"
        ]

    def choose_action(self, state):
        # Initialize a new state
        if state not in self.q_table:
            self.q_table[state] = {
                action: 0.0 for action in self.actions
            }

        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation: highest Q-value
        values = self.q_table[state]
        best_value = max(values.values())

        best_actions = [
            action
            for action, value in values.items()
            if value == best_value
        ]

        return random.choice(best_actions)

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        done=False
    ):
        # Initialize Q-values for unseen states
        for s in (state, next_state):
            if s not in self.q_table:
                self.q_table[s] = {
                    a: 0.0 for a in self.actions
                }

        old_q = self.q_table[state][action]

        # No future reward after a terminal event
        if done:
            max_next_q = 0.0
        else:
            max_next_q = max(
                self.q_table[next_state].values()
            )

        # Q-learning equation
        new_q = (
            (1 - self.learning_rate) * old_q
            + self.learning_rate * (
                reward
                + self.discount_factor * max_next_q
            )
        )

        self.q_table[state][action] = new_q

        return new_q


# This function must be OUTSIDE the class.
# It begins at the far-left indentation level.
def get_state(sensor_distances):
    state = []

    for distance in sensor_distances:
        if distance <= 30:
            category = "near"
        elif distance <= 80:
            category = "medium"
        else:
            category = "far"

        state.append(category)

    return tuple(state)
