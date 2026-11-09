
from agent import QLearningAgent

agent = QLearningAgent(epsilon=0.2)

state = ("near", "medium", "far", "far", "near")

# Example learned values
agent.q_table[state] = {
    "accelerate": 3.0,
    "brake": 1.0,
    "left": -4.0,
    "right": 9.0,
    "straight": 5.0
}

for i in range(10):
    action = agent.choose_action(state)
    print(f"Decision {i + 1}: {action}")
