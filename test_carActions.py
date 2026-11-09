
from car import Car

car = Car()

car.apply_action("accelerate_left")

print("Action: accelerate_left")
print("Steering:", car.steering)
print("Throttle:", car.throttle)

car.apply_action("brake_right")

print("\nAction: brake_right")
print("Steering:", car.steering)
print("Throttle:", car.throttle)
