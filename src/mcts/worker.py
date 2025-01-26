import time
import numpy as np
from src.mcts.mcts import MCTS
from src.mcts.node import Node
import multiprocessing as mp
from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters
from src.utils.dirichlet_noise import dirichlet_noise
from src.utils.index_to_coordinates import index_to_coordinates


class Worker(MCTS):
    def __init__(self, worker_id:int, request_queue:mp.Queue, response_queue:mp.Queue):
        super().__init__(game=OthelloGame(), model=None, root=None)

        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue

    
    def request_manager(self, state):

        self.request_queue.put((self.worker_id, state))
        while True:
            
            try:
                response = self.response_queue.get_nowait()
                if response["worker_id"] == self.worker_id:  # Check worker ID
                    action_probs, value = response["policy"], response["value"]
                    break
            except mp.queues.Empty:
                time.sleep(0.01)  # Avoid busy-waiting
        

        return action_probs, value

    
    def expand_root(self, state: np.ndarray, to_play: int, add_dirichlet_noise: bool):
        """
        Expands the root node with initial action probabilities.

        Args:
            state (np.ndarray): The initial game board state.
            to_play (int): The current player.
            add_dirichlet_noise (bool): Whether to add Dirichlet noise for exploration.
        """
        self.root.state = state
        self.root.to_play = to_play

        # Convert the board to canonical form relative to the current player.
        canonical_state = self.game.get_canonical_board(self.root.state, self.root.to_play)

        action_probs, value = self.request_manager(canonical_state)

        if add_dirichlet_noise:
            # Add Dirichlet noise to encourage exploration.
            action_probs = dirichlet_noise(action_probs)

        valid_moves = self.get_valid_moves(state, to_play)  # Get valid moves for the current state.
        action_probs = self.normalize_probs(action_probs, valid_moves)  # Normalize probabilities based on valid moves.
        self.root.expand(state, to_play, action_probs)  # Expand the root node with action probabilities.
    
    def expand_leaf(self, leaf: Node, parent: Node, action: int) -> float:
        """
        Expands a leaf node by evaluating it or determining its value.

        Args:
            leaf (Node): The leaf node to evaluate.
            parent (Node): The parent of the leaf node.
            action (int): The action leading to the leaf node.

        Returns:
            float: The evaluated value of the leaf node.
        """
        state = parent.state
        parent_player = parent.to_play
        leaf_player = parent_player * -1

        # Get the next state based on the action taken.
        (x, y) = index_to_coordinates(action)
        next_state, _ = self.game.get_next_state(state, parent_player, x, y)
        value = self.game.get_reward_for_player(next_state, leaf_player)

        if value is None:
            # Predict value and action probabilities for the next state.
            next_state_canonical = self.game.get_canonical_board(next_state, player=leaf_player)
            action_probs, value = self.request_manager(next_state_canonical)

            # Filter probabilities by valid moves and expand the leaf node.
            valid_moves = self.get_valid_moves(next_state, leaf_player)
            action_probs = self.normalize_probs(action_probs, valid_moves)
            leaf.expand(next_state, leaf_player, action_probs)

        return value

