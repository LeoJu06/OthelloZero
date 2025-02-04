"""MCTS file containing the MCTS algorithm for running search"""
import time
import numpy as np
from src.utils.mark_valid_moves import mark_valid_moves
from src.utils.index_to_coordinates import index_to_coordinates
from src.utils.dirichlet_noise import dirichlet_noise
from src.mcts.node import Node
from src.neural_net.model import OthelloZeroModel
from src.config.hyperparameters import Hyperparameters
from src.othello.othello_game import OthelloGame


class MCTS:
    """
    Monte Carlo Tree Search implementation.

    Attributes:
        game (OthelloGame): The game instance for handling game-specific logic.
        model (OthelloZeroModel): The neural network model for predicting action probabilities and values.
        hyperparameters (Hyperparameters): Configuration parameters for the MCTS.
    """

    def __init__(
        self,
        game: OthelloGame,
        model: OthelloZeroModel,
        root: Node = None,
    ):
        """
        Initializes the MCTS instance with the game, model, and optional root node.

        Args:
            game (OthelloGame): Instance of the game being played.
            model (OthelloZeroModel): Neural network model for predictions.
            root (Node, optional): Root node for the MCTS tree. Defaults to None.
        """
        self.game = game
        self.model = model
        self.hyperparameters = Hyperparameters()
        self.root = Node(prior=0, to_play=-1)  # Initialize root with default values.

    def run(
        self, state: np.ndarray, to_play: int, add_dirichlet_noise: bool = True
    ) -> Node:
        """
        Executes MCTS to determine the optimal policy and value.

        Args:
            state (np.ndarray): The initial game board state.
            to_play (int): The current player.
            add_dirichlet_noise (bool): Whether to add Dirichlet noise for exploration. Defaults to True.

        Returns:
            Node: The updated root node after simulations.
        """
        # Expand the root node with initial probabilities.
        self.expand_root(state, to_play, add_dirichlet_noise)

        # Perform a series of simulations.
        for _ in range(self.hyperparameters.MCTS["num_simulations"]):
            search_path, action_path = self.tree_traverse(
                self.root
            )  # Traverse tree to a leaf.
            value = self.expand_leaf(
                search_path[-1], search_path[-2], action_path[-1]
            )  # Expand the leaf.
            self.backpropagate(
                search_path, value
            )  # Backpropagate the evaluation value.

        return self.root

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
        canonical_state = self.game.get_canonical_board(
            self.root.state, self.root.to_play
        )
        action_probs, _ = self.model.predict(canonical_state)

        if add_dirichlet_noise:
            # Add Dirichlet noise to encourage exploration.
            action_probs = dirichlet_noise(action_probs)

        valid_moves = self.get_valid_moves(
            state, to_play
        )  # Get valid moves for the current state.

        action_probs = self.normalize_probs(
            action_probs, valid_moves
        )  # Normalize probabilities based on valid moves.

        self.root.expand(
            state, to_play, action_probs
        )  # Expand the root node with action probabilities.

    def tree_traverse(self, node: Node) -> list:
        """
        Traverses the tree by selecting child nodes based on the highest action score.

        Args:
            node (Node): The current node to start traversing from.

        Returns:
            list: The search path and the corresponding actions taken during traversal.
        """
        search_path = [node]
        action_path = []

        while node.expanded():
            # Select the best child node and corresponding action.
            action, node = node.select_child()
            search_path.append(node)
            action_path.append(action)

        return search_path, action_path

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
            next_state_canonical = self.game.get_canonical_board(
                next_state, player=leaf_player
            )

            start = time.time()
            action_probs, value = self.model.predict(next_state_canonical)
            # print(f"MCTS waited {time.time()-start:4f} seconds for a response")

            # Filter probabilities by valid moves and expand the leaf node.
            valid_moves = self.get_valid_moves(next_state, leaf_player)
            action_probs = self.normalize_probs(action_probs, valid_moves)
            leaf.expand(next_state, leaf_player, action_probs)

        return value

    def get_valid_moves(self, state: np.ndarray, to_play: int) -> np.ndarray:
        """
        Retrieves valid moves for a given state and player.

        Args:
            state (np.ndarray): The board state.
            to_play (int): The current player.

        Returns:
            np.ndarray: Flattened array of valid moves.
        """
        return self.game.flatten_move_coordinates(state, to_play)

    def normalize_probs(
        self, action_probs: np.ndarray, valid_moves: np.ndarray
    ) -> np.ndarray:
        """
        Normalizes action probabilities based on valid moves.

        Args:
            action_probs (np.ndarray): Raw action probabilities.
            valid_moves (np.ndarray): Valid moves for the current state.

        Returns:
            np.ndarray: Normalized probabilities.
        """
        action_probs = mark_valid_moves(action_probs, valid_moves)
        return action_probs / action_probs.sum()

    def backpropagate(self, search_path: list, value: float):
        """
        Backpropagates the evaluation value up the search path.

        Args:
            search_path (list): List of nodes in the search path.
            value (float): The evaluation value to propagate.
        """
        to_play = search_path[-1].to_play

        for node in reversed(search_path):
            # Update value sum and visit count for each node.
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1


def dummy_console_mcts(args):
    """
    Runs a dummy MCTS example using the console for testing.
    """
    model, process_id = args
    print(f"Process {process_id}: Model memory address = {id(model)}")
    import time

    # Initialize hyperparameters, game, and model

    g = OthelloGame()
    s = g.get_init_board()
    current_player = -1

    t = []

    # Run MCTS loop until the game reaches a terminal state.
    while not g.is_terminal_state(s):
        start_time = time.time()
        mcts = MCTS(g, model)
        r = mcts.run(s, current_player)
        a = r.select_action(temperature=0)
        x, y = index_to_coordinates(a)
        s, current_player = g.get_next_state(s, current_player, x, y)

        g.print_board(s)  # Display the current board.
        print(f"Move played= {x}, {y}")
        tn = time.time() - start_time
        print(f"Thinking time {tn:.2f} seconds")
        t.append(tn)

    # g.print_board(s)  # Display final board state.
    print(f"Average thinking time {sum(t)/len(t):.4f} seconds")
    print(f"Total lenght of game = {sum(t):2f} seconds")

    return mcts.root.visit_count


if __name__ == "__main__":
    import multiprocessing
    import torch.multiprocessing as tmp

    num_processes = 4  # Anzahl der parallelen MCTS-Instanzen
    h = Hyperparameters()
    g = OthelloGame()
    s = g.get_init_board()
    current_player = -1

    tmp.set_start_method("spawn", force=True)

    model = OthelloZeroModel(g.rows, g.get_action_size(), h.Neural_Network["device"])
    model.eval()
    model.share_memory()
    args_list = [(model, i) for i in range(num_processes)]
    with multiprocessing.Pool(num_processes) as pool:
        result = pool.map(dummy_console_mcts, args_list)

    print(result)
