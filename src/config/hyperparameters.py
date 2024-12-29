import torch


class Hyperparameters:
    MCTS = {"num_simulations": 10000, "exploration_weight": 1.0}

    Neural_Network = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    Node = {"key_passing": -1, "prior_passing": 1}
