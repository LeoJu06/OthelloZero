import multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
import time
import src.utils.logger_config as lg
from src.config.hyperparameters import Hyperparameters
from src.mcts.node import Node
from src.neural_net.model import neural_network_evaluate, NeuralNetwork


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation with multiprocessing capabilities.
    """

    def __init__(self, worker_id=None, root_node=None):
        """
        Initialize the MCTS object.

        Args:
            worker_id (int, optional): Unique identifier for the worker.
            root_node (Node, optional): The root node of the MCTS tree. If not provided, a new root node is created.
        """
        self.worker_id = worker_id  # Unique ID for the worker, used to track its tasks and responses.
        if root_node is None:
            self.root = Node(
                prior=float("inf")
            )  # Initialize the root node with infinite prior.
        else:
            self.root = root_node  # Use the provided root node.
        pass  # Explicitly indicating that no other initialization is required.

    def get_best_move(self):
        """Method to return the best move with the correspondending child"""

        best_move = None
        best_child = None
        max_visits = float("-inf")

        for move, child in self.root.children.items():

            if child.visits > max_visits:

                best_move = move
                best_child = child

        return best_move, best_child





    