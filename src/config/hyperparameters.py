"Module to hold the Hyperparamter class."
import torch


class Hyperparameters:

    """The Hyperparameter class contains dictionnaries with
    hyperparameters needed for certain tasks.
    Simply address them by writing Hyperparamerts.Name["key"]

    The Hyperparameter class contains:
        - MCTS with keys ["num_simulations", "exploration_weight"]
        - Neural_Network with keys ["device"]
        - Node with keys ["key_passsing, prior_passing]"""

    MCTS = {
        "num_simulations": 700,
        "exploration_weight": 1.0,
        "temperature_turn_threshold": 1,
        "temperature": 1,
    }

    Neural_Network = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    Node = {"key_passing": -1, "prior_passing": 1}
