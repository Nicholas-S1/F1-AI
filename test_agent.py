
from agent import get_state

# Simulated sensor readings
distances = [20, 45, 150, 100, 30]

state = get_state(distances)

print("Sensor distances:", distances)
print("AI state:", state)
