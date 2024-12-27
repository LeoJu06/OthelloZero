from src.othello.board import Board
from src.utils.index_to_coordinates import index_to_coordinates
from src.neural_net.model import dummy_model_predict
from src.config.hyperparameters import Hyperparameters
import numpy as np
import time
import math


def ucb_score(
    parent, child, exploration_weight=Hyperparameters.MCTS["exploration_weight"]
):
    """
    Computes the Upper Confidence Bound (UCB) score for a child node.

    The UCB score is calculated using the following formula:
    UCB = value_score + exploration_weight * prior_score
    where:
        - value_score = the average value of the child node based on the number of visits
        - prior_score = the prior probability of selecting the child (scaled by the parent node's visits)

    Args:
        parent (Node): The parent node.
        child (Node): The child node.
        exploration_weight (float): A weight factor for the exploration term (default from Hyperparameters).

    Returns:
        float: The UCB score for the child node.
    """
    # Calculate the prior score
    prior_score = child.prior * math.sqrt(parent.visits) / (child.visits + 1)

    # Calculate the value score
    value_score = (child.value / child.visits) if child.visits > 0 else 0

    return value_score + exploration_weight * prior_score


class Node:
    """
    A Node in the Monte Carlo Tree Search (MCTS).

    Each node represents a state in the game tree. It contains information about:
        - The prior probability of the node (based on action probabilities)
        - The current board state
        - Child nodes
        - The accumulated value and number of visits

    Attributes:
        prior (float): The prior probability of selecting the node.
        board (Board): The current state of the game board.
        children (dict): A dictionary of child nodes, indexed by the action taken.
        value (float): The accumulated value (e.g., win/loss) for the node.
        visits (int): The number of visits (i.e., simulations) that have passed through this node.
    """

    def __init__(self, prior, board=None):
        """
        Initializes a new node in the MCTS.

        Args:
            prior (float): The prior probability of selecting the node.
            board (Board, optional): The current state of the board. If None, a new Board is created.
        """
        self.prior = prior  # Probability of playing this move
        if board is None:
            self.board = Board()  # Create a new Board instance if none is passed
        else:
            self.board = board  # If a board is passed, assign it directly

        self.children = {}  # Dictionary of children nodes
        self.value = 0  # Accumulated value of the node
        self.visits = 0  # Number of visits to the node

    def expand(self, action_probs):
        """
        Expands the node by creating child nodes for valid moves based on action probabilities.

        For each valid move, a new child node is created with a corresponding prior probability.

        Args:
            action_probs (list of float): The action probabilities (prior probabilities for each valid move).
        """
        valid_moves = (
            self.board.valid_moves()
        )  # Get the valid moves for the current board state
        for action, prob in enumerate(action_probs):
            if prob > 0:  # Only consider actions with a non-zero probability
                x, y = index_to_coordinates(
                    action
                )  # Convert action index to coordinates
                if (x, y) in valid_moves:  # Check if the move is valid
                    child_board = Board(
                        board=np.copy(self.board.board)
                    )  # Copy the current board
                    child_board.apply_move(x, y)  # Apply the move to the child board
                    child_board.update(x, y)  # Update the board state

                    # Create a new child node for the selected move
                    child = Node(prior=prob, board=child_board)
                    self.children[
                        action
                    ] = child  # Add the child node to the children dictionary

    def select_child(self):
        """
        Selects a child node based on the UCB score.

        The child with the highest UCB score is selected.

        Returns:
            tuple: The selected action and the corresponding child node.
        """
        max_score = float("-inf")  # Initialize the max score to negative infinity
        selected_action = None
        selected_child = None

        # Iterate over each child and compute the UCB score
        for action, child in self.children.items():
            score = ucb_score(self, child)  # Compute the UCB score for the child node
            if score > max_score:
                max_score = score
                selected_action = (
                    action  # Select the action corresponding to the highest score
                )
                selected_child = child  # Select the corresponding child node

        return (
            selected_action,
            selected_child,
        )  # Return the selected action and child node

    def is_expanded(self):
        """
        Checks if the node has been expanded (i.e., if it has child nodes).

        Returns:
            bool: True if the node has child nodes, False otherwise.
        """
        return len(self.children) > 0

    def is_terminal_state(self):
        """
        Checks if the current board state is a terminal state (e.g., win, loss, or draw).

        Returns:
            bool: True if the current board state is terminal, False otherwise.
        """
        return self.board.is_terminal_state()


if __name__ == "__main__":
    # Main MCTS Simulation
    num_simulations = 1200  # Number of MCTS simulations to run
    start = time.time()  # Record the start time for performance measurement
    root = Node(
        prior=float("inf")
    )  # Create the root node with an infinite prior (starting point)

    # Expand the root node using the action probabilities from the model
    action_probs, _ = dummy_model_predict(root.board)
    root.expand(action_probs)

    # Run the MCTS simulations
    for _ in range(num_simulations):
        node = root
        search_path = [node]  # Keep track of the nodes visited in this simulation

        # Selection phase: Traverse the tree to select a node to expand
        while node.is_expanded():
            (
                action,
                node,
            ) = node.select_child()  # Select the child node with the highest UCB score
            search_path.append(node)

        # Evaluation phase: Evaluate the value of the node (either terminal or predicted)
        value = None
        if node.is_terminal_state():
            value = node.board.determine_winner()  # If terminal, use the winner value
        if value is None:
            action_probs, value = dummy_model_predict(
                node.board
            )  # Predict value using the model if not terminal
            node.expand(
                action_probs
            )  # Expand the node with the new action probabilities

        # Backpropagation phase: Update the value and visit count of the nodes along the search path
        for node in search_path:
            node.value += value  # Update node value based on simulation outcome
            node.visits += 1  # Increment the number of visits to this node

    # Simulation results: Print the results of the MCTS simulations
    print(f"Root's value => {root.value}")
    print(f"Root's visits => {root.visits}")
    print(
        f"Time needed for {num_simulations} iterations => {time.time() - start:.2f} seconds"
    )

    # Print details for each move
    for move, child in root.children.items():
        print(
            f"Move => {move}, Visits => {child.visits}, Value => {child.value:.2f}, UCB Score => {ucb_score(root, child):.2f}"
        )
