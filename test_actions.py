
from agent import QLearningAgent

agent = QLearningAgent(epsilon=0.2)

state = ("near", "medium", "far", "far", "near")

# Example learned values (uses the agent's real action names)
agent.q_table[state] = {
    "accelerate_straight": 5.0,
    "accelerate_left": -4.0,
    "accelerate_right": 9.0,
    "brake_straight": 1.0,
    "brake_left": -2.0,
    "brake_right": 3.0,
    "coast_straight": 2.0,
}

# With epsilon=0.2, most picks should be "accelerate_right"
for i in range(10):
    action = agent.choose_action(state)
    print(f"Decision {i + 1}: {action}")
