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
        "num_simulations": 100,
        "exploration_weight": 1.0,
        "temp_threshold": 10,
        "temp": 1,
    }

    Coach = {"iterations": 2, 
             "episodes": 2500,
             "num_workers" :22, }
    Coach["episodes_per_worker"] = Coach["episodes"] // Coach["num_workers"]

    Neural_Network = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    Node = {"key_passing": -1, "prior_passing": 1}
