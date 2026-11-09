
from agent import QLearningAgent
from memory import save_memory, load_memory, reset_memory

agent = QLearningAgent()

state = ("near", "far", "far", "medium", "near")

# Create some example learned values
agent.q_table[state] = {
    action: 0.0 for action in agent.actions
}

agent.q_table[state]["accelerate_left"] = 8.5

# Save the AI
save_memory(agent)

# Clear the AI's memory
reset_memory(agent)

print("States after reset:", len(agent.q_table))

# Load the saved AI
load_memory(agent)

print("States after loading:", len(agent.q_table))
print(
    "Restored Q-value:",
    agent.q_table[state]["accelerate_left"]
)
