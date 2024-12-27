from src.othello.board import Board
from src.utils.index_to_coordinates import index_to_coordinates
from src.neural_net.model import dummy_model_predict
import numpy as np
import random
import time
import math



def ucb_score(parent, child, exploration_weight=1.0):
    """
    Computes the Upper Confidence Bound (UCB) score for a child node.
    """
    prior_score = child.prior * math.sqrt(parent.visits) / (child.visits + 1)
    value_score = (child.value / child.visits) if child.visits > 0 else 0
    return value_score + exploration_weight * prior_score

class Node:
    """
    A Node in the Monte Carlo Tree Search.
    """

    def __init__(self, prior, board=None):
        self.prior = prior  # Probability of playing this move
        if board is None:
            self.board = Board()  # Create a new Board instance if none is passed
        else:
            self.board = board  # If a board is passed, assign it directly

        self.children = {}  # Dictionary of children nodes
        self.value = 0      # Accumulated value of the node
        self.visits = 0     # Number of visits to the node

    def expand(self, action_probs):
        """
        Expands the node by creating child nodes for valid moves.
        """
        valid_moves = self.board.valid_moves()
        for action, prob in enumerate(action_probs):
            if prob > 0:
                x, y = index_to_coordinates(action)
                if (x, y) in valid_moves:
                    child_board = Board(board=np.copy(self.board.board))
                    child_board.apply_move(x, y)
                    child_board.update(x, y)

                    child = Node(prior=prob, board=child_board)
                    self.children[action] = child

    def select_child(self):
        """
        Selects a child node based on UCB score.
        """
        max_score = float("-inf")
        selected_action = None
        selected_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > max_score:
                max_score = score
                selected_action = action
                selected_child = child

        return selected_action, selected_child

    def is_expanded(self):
        """
        Checks if the node has been expanded (i.e., has children).
        """
        return len(self.children) > 0

    def is_terminal_state(self):
        """
        Checks if the current board state is a terminal state.
        """
        return self.board.is_terminal_state()


if __name__ == "__main__":

    # Main MCTS Simulation
    num_simulations = 1200
    start = time.time()
    root = Node(prior=float("inf"))

    # Expand the root node
    action_probs, _ = dummy_model_predict(root.board)
    root.expand(action_probs)
    

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # Selection phase
        while node.is_expanded():
            action, node = node.select_child()
            search_path.append(node)

        # Evaluation phase
        value = None
        if node.is_terminal_state():
            value = node.board.determine_winner()
        if value is None:
            action_probs, value = dummy_model_predict(node.board)
            node.expand(action_probs)

        # Backpropagation phase
        for node in search_path:
            node.value += value
            node.visits += 1

    # Simulation results
    print(f"Root's value => {root.value}")
    print(f"Root's visits => {root.visits}")
    print(f"Time needed for {num_simulations} iterations => {time.time() - start:.2f} seconds")

    for move, child in root.children.items():
        print(f"Move => {move}, Visits => {child.visits}, Value => {child.value:.2f}, UCB Score => {ucb_score(root, child):.2f}")
