
from agent import QLearningAgent

agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.2
)

state = ("near", "medium", "far", "far", "near")
next_state = ("medium", "medium", "far", "far", "near")

# Example Q-values
agent.q_table[state] = {
    "accelerate": 0.0,
    "brake": 0.0,
    "left": 0.0,
    "right": 2.0,
    "straight": 0.0
}

agent.q_table[next_state] = {
    "accelerate": 3.0,
    "brake": 1.0,
    "left": 0.0,
    "right": 5.0,
    "straight": 2.0
}

# The agent took the "right" action and earned reward 10
new_value = agent.learn(
    state=state,
    action="right",
    reward=10,
    next_state=next_state,
    done=False
)

print("Updated Q-value:", new_value)
