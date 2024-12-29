from src.othello.board import Board
from src.utils.index_to_coordinates import index_to_coordinates
from src.neural_net.model import dummy_model_predict
from src.config.hyperparameters import Hyperparameters
import numpy as np
import time
import math
from src.utils.print_green import print_green

def ucb_score(parent, child, exploration_weight=Hyperparameters.MCTS["exploration_weight"]):
    """
    Computes the Upper Confidence Bound (UCB) score for a child node.

    UCB = value_score + exploration_weight * prior_score
    - value_score: The average value of the child node based on the number of visits.
    - prior_score: The prior probability of selecting the child, scaled by parent node visits.

    Args:
        parent (Node): The parent node.
        child (Node): The child node.
        exploration_weight (float): Weight factor for the exploration term.

    Returns:
        float: The UCB score for the child node.
    """
    prior_score = child.prior * math.sqrt(parent.visits) / (child.visits + 1)
    value_score = (child.value / child.visits) if child.visits > 0 else 0
    return value_score + exploration_weight * prior_score


class Node:
    """
    Represents a state in the Monte Carlo Tree Search (MCTS).

    Attributes:
        prior (float): The prior probability of selecting this node.
        board (Board): The current game board state.
        children (dict): Dictionary mapping actions to child nodes.
        value (float): Accumulated value from simulations.
        visits (int): Number of visits to this node.
    """

    def __init__(self, prior=float("inf"), board=None):
        """
        Initializes a new node for MCTS.

        Args:
            prior (float): Prior probability of selecting this node.
            board (Board, optional): Current game state. Defaults to a new `Board` instance.
        """
        self.prior = prior
        self.board = board if board else Board()
        self.children = {}
        self.value = 0
        self.visits = 0

    def expand(self, action_probs):
        """
        Expands the node by creating child nodes for valid moves.
        If the current player has no valid moves, skips their turn.

        Args:
            action_probs (list[float]): Prior probabilities for each action.
        """
        valid_moves = self.board.valid_moves()

        if not valid_moves:
            self._expand_pass_node()
            return

        self._expand_valid_moves(action_probs, valid_moves)

    def _expand_pass_node(self):
        """Handles the case where the current player must pass."""
        child_board = Board(
            board=np.copy(self.board.board), player=self.board.player
        )
        child_board.update()  # Switch to the opponent's turn
        child = Node(prior=1.0, board=child_board)  # Default prior for passing
        self.children[-1] = child

    def _expand_valid_moves(self, action_probs, valid_moves):
        """
        Expands the node with valid moves.

        Args:
            action_probs (list[float]): Prior probabilities for actions.
            valid_moves (list[tuple]): List of valid moves as (x, y) coordinates.
        """
        for action, prob in enumerate(action_probs):
            if prob > 0:
                x, y = index_to_coordinates(action)
                if (x, y) in valid_moves:
                    self._add_child_node(action, prob, x, y)

    def _add_child_node(self, action, prob, x, y):
        """
        Adds a child node for a specific action.

        Args:
            action (int): Action index.
            prob (float): Prior probability of the action.
            x (int): X-coordinate of the move.
            y (int): Y-coordinate of the move.
        """
        child_board = Board(
            board=np.copy(self.board.board), player=self.board.player
        )
        child_board.apply_move(x, y)
        child_board.update(x, y)
        child = Node(prior=prob, board=child_board)
        self.children[action] = child

    def select_child(self):
        """
        Selects the child node with the highest UCB score.

        Returns:
            tuple: (selected_action, selected_child)
        """
        return max(
            self.children.items(),
            key=lambda item: ucb_score(self, item[1])
        )

    def is_expanded(self):
        """
        Checks if the node has been expanded (i.e., has children).

        Returns:
            bool: True if the node has children, False otherwise.
        """
        return bool(self.children)

    def is_terminal_state(self):
        """
        Checks if the current board state is terminal.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return self.board.is_terminal_state()
