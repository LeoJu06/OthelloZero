"Module to hold the Hyperparamter class."
import torch


class Hyperparameters:

    """The Hyperparameter class contains dictionnaries with
    hyperparameters needed for certain tasks.
    Simply address them by writing Hyperparamerts.Name["key"]

    The Hyperparameter class contains:
        - MCTS with keys ["num_simulations", "exploration_weight", "temp_threshold", temperature]
        - Coach with keys ["iterations", "episodes", "num_workers", "episodes_per_worker"]
        - Neural_Network with keys ["device"]
        - Node with keys ["key_passsing, prior_passing]"""

    MCTS = {
        "num_simulations": 10,
        "exploration_weight": 1.5,
        "temp_threshold": 14,
        "temp": 1,
        "data_turn_limit": 55
    }

    Coach = {"iterations": 50, 
             "episodes": 22*30,
             "num_workers" :22,
              "arena_competition": 5}
    Coach["episodes_per_worker"] = Coach["episodes"] // Coach["num_workers"]

    Neural_Network = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        "epochs" : 10, 
        "batch_size" : 128,
        "learning_rate" : 0.01
    }

    Node = {"key_passing": -1, "prior_passing": 1}

    Arena = {"treshold": 0.55, "arena_games": 150}


def temperature(move_num, total_moves=60):
    return max(0.1, 1.0 - move_num*0.03)  # Linear von 1.0 → 0.1 über 30 Züge

def mcts_simulations(iteration):
    if iteration <= 5:
        return 100
    elif iteration <= 15:
        return 200
    elif iteration <= 30:
        return 300
    elif iteration <= 50:
        return 400
    else:
        return 600

    