import time
import numpy as np
from src.mcts.mcts import MCTS
from src.mcts.node import Node
import torch
import multiprocessing as mp
from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters
from src.utils.dirichlet_noise import dirichlet_noise
from src.utils.index_to_coordinates import index_to_coordinates

class Worker(MCTS):
    def __init__(self, worker_id, request_queue, shared_states, shared_policies, shared_values, response_event):
        super().__init__(game=OthelloGame(), model=None, root=None)
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.shared_states = shared_states  # Shared memory for states
        self.shared_policies = shared_policies  # Shared memory for policies
        self.shared_values = shared_values  # Shared memory for values
        self.response_event = response_event  # Event for synchronization


    def request_manager(self, state):
        """Writes the state to shared memory and waits for a response."""
        self.shared_states[self.worker_id] = torch.tensor(state, dtype=torch.float32)

        # Send request to manager
        self.request_queue.put((self.worker_id, self.worker_id))

        # Wait for manager's response
        self.response_event.wait()
        self.response_event.clear()  # Reset event for next request

        # Read response from shared memory
        return self.shared_policies[self.worker_id], self.shared_values[self.worker_id]

    def expand_root(self, state: np.ndarray, to_play: int, add_dirichlet_noise: bool):
        """Expands the root node with initial action probabilities."""
        self.root.state = state
        self.root.to_play = to_play

        # Convert the board to canonical form relative to the current player.
        canonical_state = self.game.get_canonical_board(self.root.state, self.root.to_play)

        action_probs, value = self.request_manager(canonical_state)

        if add_dirichlet_noise:
            action_probs = dirichlet_noise(action_probs)

        valid_moves = self.get_valid_moves(state, to_play)  
        action_probs = self.normalize_probs(action_probs, valid_moves)  
        self.root.expand(state, to_play, action_probs)  
    
    def expand_leaf(self, leaf: Node, parent: Node, action: int) -> float:
        """Expands a leaf node by evaluating it or determining its value."""
        state = parent.state
        parent_player = parent.to_play
        leaf_player = parent_player * -1

        (x, y) = index_to_coordinates(action)
        next_state, _ = self.game.get_next_state(state, parent_player, x, y)
        value = self.game.get_reward_for_player(next_state, leaf_player)

        if value is None:
            next_state_canonical = self.game.get_canonical_board(next_state, player=leaf_player)
            action_probs, value = self.request_manager(next_state_canonical)

            valid_moves = self.get_valid_moves(next_state, leaf_player)
            action_probs = self.normalize_probs(action_probs, valid_moves)
            leaf.expand(next_state, leaf_player, action_probs)

        return value
