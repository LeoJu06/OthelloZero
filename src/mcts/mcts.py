"""MCTS file containing the MCTS algorithm for running search"""
import numpy as np
from src.utils.mark_valid_moves import mark_valid_moves
from src.utils.index_to_coordinates import index_to_coordinates
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
        hyperparameters: Hyperparameters,
        root: Node = None,
    ):
        self.game = game
        self.model = model
        self.hyperparameters = hyperparameters
        self.root = Node(prior=0, to_play=-1)

    

    def run(self, state: np.ndarray, to_play: int) -> Node:
        """
        Executes MCTS to determine the optimal policy and value.

        Args:
            state (np.ndarray): The initial game board state.
            to_play (int): The current player.

        Returns:
            Node: The updated root node after simulations.
        """
        self.expand_root(state, to_play)

        for _ in range(self.hyperparameters.MCTS["num_simulations"]):
            search_path = self.tree_traverse(self.root)
            value = self.expand_leaf(search_path[-1], search_path[-2])
            self.backpropagate(search_path, value)

        return self.root

    def expand_root(self, state: np.ndarray, to_play: int):
        """
        Expands the root node with initial action probabilities.

        Args:
            state (np.ndarray): The initial game board state.
            to_play (int): The current player.
        """
        action_probs, _ = self.model.predict(state)
        valid_moves = self.get_valid_moves(state, to_play)
        action_probs = self.normalize_probs(action_probs, valid_moves)
        self.root.expand(state, to_play, action_probs)

    def tree_traverse(self, node: Node) -> list:
        """
        Traversing the tree by selecting child nodes.

        Args:
            node (Node): The current node to start the traversing.

        Returns:
            list: The search path taken during tree traversing.
        """
        search_path = [node]

        while node.expanded():
            _, node = node.select_child()
            search_path.append(node)

        return search_path

    def expand_leaf(self, leaf: Node, parent: Node) -> float:
        """
        Expanding a leaf node by evaluating it or determining its value.

        Args:
            leaf (Node): The leaf node to evaluate.
            parent (Node): The parent of the leaf node.

        Returns:
            float: The evaluated value of the leaf node.
        """
        state = parent.state
        action_probs, value = self.model.predict(state)
        valid_moves = self.get_valid_moves(state, parent.to_play * -1)
        action_probs = self.normalize_probs(action_probs, valid_moves)
        leaf.expand(state, parent.to_play * -1, action_probs)
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
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

def dummy_console_mcts():
    import time
    # Initialize hyperparameters, game, and model
    h = Hyperparameters()
    g = OthelloGame()
    s = g.get_init_board()
    current_player = -1
    m = OthelloZeroModel(g.rows, g.get_action_size(), h.Neural_Network["device"])
    t = []
    # Run MCTS
    while not g.is_terminal_state(s):
        start_time = time.time()
        mcts = MCTS(g, m, h)
        r = mcts.run(s, current_player)
        a = r.select_action(temperature=0)
        x, y = index_to_coordinates(a)
        s, current_player = g.get_next_state(s, current_player,x, y )

        g.print_board(s)
        print(f"Move played= {x}, {y}")
        tn = time.time() - start_time
        print(f"Thinking time {tn:.2f} sconds")
        t.append(tn)
    
    g.print_board(s)
    print(f"Average thinking time {sum(t)/len(t):.4f} seconds")



if __name__ == "__main__":
    
    dummy_console_mcts()
        
