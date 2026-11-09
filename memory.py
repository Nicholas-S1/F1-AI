
import os
import pickle

MODEL_FOLDER = "models"
MODEL_FILE = os.path.join(
    MODEL_FOLDER, "f1_ai.pkl"
)


def save_memory(agent):
    """Save the AI's learned Q-values."""

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    data = {
        "q_table": agent.q_table,
        "epsilon": agent.epsilon,
        "learning_rate": agent.learning_rate,
        "discount_factor": agent.discount_factor,
        "actions": agent.actions
    }

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(data, file)

    print("AI memory saved successfully!")


def load_memory(agent):
    """Restore previously saved AI memory."""

    if not os.path.exists(MODEL_FILE):
        print("No saved memory found.")
        return False

    with open(MODEL_FILE, "rb") as file:
        data = pickle.load(file)

    agent.q_table = data["q_table"]
    agent.epsilon = data["epsilon"]
    agent.learning_rate = data["learning_rate"]
    agent.discount_factor = data["discount_factor"]
    agent.actions = data["actions"]

    print("AI memory loaded successfully!")
    return True


def reset_memory(agent):
    """Clear the AI's learned memory."""

    agent.q_table.clear()
    agent.epsilon = 1.0

    print("AI memory reset!")
