
import os
import pickle

MODEL_FOLDER = "models"
MODEL_FILE = os.path.join(
    MODEL_FOLDER, "f1_ai.pkl"
)


def _as_list(agents):
    return agents if isinstance(agents, (list, tuple)) else [agents]


def _agent_data(agent, extra):
    data = {
        "q_table": agent.q_table,
        "epsilon": agent.epsilon,
        "learning_rate": agent.learning_rate,
        "discount_factor": agent.discount_factor,
        "actions": agent.actions,
    }
    data.update(extra or {})
    return data


def _apply_agent_data(agent, data):
    # Copy the table so cars loaded from the same
    # saved brain still learn independently
    agent.q_table = {
        state: dict(values)
        for state, values in data["q_table"].items()
    }
    agent.epsilon = data["epsilon"]
    agent.learning_rate = data["learning_rate"]
    agent.discount_factor = data["discount_factor"]
    agent.actions = list(data["actions"])


def save_memory(agents, extras=None):
    """
    Save the learned Q-values of one agent or a list
    of agents. `extras` is an optional list of dicts
    (one per agent) of other settings to store, such
    as each car's top speed.
    """
    agents = _as_list(agents)
    extras = extras or [{} for _ in agents]

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    data = {
        "agents": [
            _agent_data(agent, extra)
            for agent, extra in zip(agents, extras)
        ]
    }

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(data, file)

    print("AI memory saved successfully!")


def load_memory(agents):
    """
    Restore previously saved AI memory into one agent
    or a list of agents.

    Returns False if nothing was saved, otherwise a list
    with each agent's saved extras (e.g. top speed).

    Files saved with a single agent (older versions) are
    copied into every agent. If the file has fewer agents
    than requested, the saved ones are reused in order.
    """
    agents = _as_list(agents)

    if not os.path.exists(MODEL_FILE):
        print("No saved memory found.")
        return False

    with open(MODEL_FILE, "rb") as file:
        data = pickle.load(file)

    if "agents" in data:
        saved = data["agents"]
    else:
        saved = [data]

    extras = []

    for i, agent in enumerate(agents):
        entry = saved[i % len(saved)]
        _apply_agent_data(agent, entry)
        extras.append({
            key: value for key, value in entry.items()
            if key not in (
                "q_table", "epsilon", "learning_rate",
                "discount_factor", "actions",
            )
        })

    print("AI memory loaded successfully!")
    return extras


def reset_memory(agents):
    """Clear the AI's learned memory."""

    for agent in _as_list(agents):
        agent.q_table.clear()
        agent.epsilon = 1.0

    print("AI memory reset!")
