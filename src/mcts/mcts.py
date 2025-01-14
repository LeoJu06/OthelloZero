import math
import numpy as np
from src.utils.mark_valid_moves import mark_valid_moves
from src.utils.index_to_coordinates import index_to_coordinates
from src.neural_net.model import OthelloZeroModel
from src.config.hyperparameters import Hyperparameters
from src.othello.othello_game import OthelloGame


def ucb_score(parent, child):
    """
    Calculates the Upper Confidence Bound (UCB) score for a given child node.

    Args:
        parent (Node): The parent node.
        child (Node): The child node.

    Returns:
        float: The UCB score for the child node.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    """
    Represents a single node in the MCTS tree.

    Attributes:
        visit_count (int): Number of times this node has been visited.
        to_play (int): The player whose turn it is to play at this node.
        prior (float): The prior probability of selecting this node.
        value_sum (float): The cumulative value of this node.
        children (dict): Dictionary of child nodes keyed by actions.
        state (np.ndarray): The board state associated with this node.
    """

    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        """
        Checks if the node has been expanded (i.e., has children).

        Returns:
            bool: True if the node has children, False otherwise.
        """
        return len(self.children) > 0

    def value(self):
        """
        Calculates the average value of the node.

        Returns:
            float: The average value, or 0 if the node has not been visited.
        """
        if self.visit_count == 0:
            return 0
        return float(self.value_sum / self.visit_count)

    def select_action(self, temperature):
        """
        Selects an action based on the visit count distribution and a temperature parameter.

        Args:
            temperature (float): Determines the level of exploration (0 for greedy, inf for random).

        Returns:
            int: The selected action.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # Adjust distribution based on temperature
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Selects the child with the highest UCB score.

        Returns:
            tuple: The best action and the corresponding child node.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        Expands the node by creating child nodes for each valid action.

        Args:
            state (np.ndarray): The board state at this node.
            to_play (int): The player to play at this node.
            action_probs (np.ndarray): The prior probabilities for each action.
        """
        self.to_play = to_play
        self.state = state

        # Handle the case where no valid moves are available
        if action_probs[-1] != 0:
            
            # No valid moves, so the player must pass
            self.children[-1] = Node(prior=action_probs[-1], to_play=self.to_play * -1)
            return

        for move, prob in enumerate(action_probs):
            if prob != 0:
                self.children[move] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self):
        """
        Provides a string representation of the node for debugging purposes.

        Returns:
            str: A formatted string representing the node.
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(
            self.state.__str__(), prior, self.visit_count, self.value()
        )


class MCTS:
    """
    Monte Carlo Tree Search implementation.

    Attributes:
        game (OthelloGame): The game instance for handling game-specific logic.
        model (OthelloZeroModel): The neural network model for predicting action probabilities and values.
        hyperparameters (Hyperparameters): Configuration parameters for the MCTS.
    """

    def __init__(self, game: OthelloGame, model, hyperparameters, root: Node = None):
        self.game = game
        self.model = model
        self.hyperparameters = hyperparameters
        self.root = root if root else Node(0, -1)

    def run(self, state, to_play):
        """
        Runs the MCTS algorithm to compute the optimal policy and value for a given state.

        Args:
            state (np.ndarray): The initial board state.
            to_play (int): The player to play.

        Returns:
            Node: The root node of the search tree after simulations.
        """
        root = self.root

        # EXPAND root
        action_probs, value = self.model.predict(state)
        valid_moves_flattened = self.game.flatten_move_coordinates(state, to_play)
        action_probs = mark_valid_moves(action_probs, valid_moves_flattened)
        action_probs /= np.sum(action_probs)
        root.expand(state, to_play, action_probs)

        for _ in range(self.hyperparameters.MCTS["num_simulations"]):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state

            # Check for valid moves
            valid_moves_flattened = self.game.flatten_move_coordinates(
                state, parent.to_play * -1
            )
            # print(valid_moves_flattened)
            # print(action)
            if action == -1:
                # No valid moves, player must pass
                next_state = self.game.get_canonical_board(
                    state, player=parent.to_play * -1
                )
                action_probs, value = self.model.predict(next_state)
                action_probs = mark_valid_moves(action_probs, valid_moves_flattened)
                # print(action_probs)
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.to_play * -1, action_probs)
            else:
                # Now we're at a leaf node and we would like to expand
                (x, y) = index_to_coordinates(action)
                next_state, _ = self.game.get_next_state(
                    state, player=1, x_pos=x, y_pos=y
                )
                next_state = self.game.get_canonical_board(next_state, player=-1)

                # Get the reward or expand further
                value = self.game.get_reward_for_player(next_state, player=1)
                if value is None:
                    action_probs, value = self.model.predict(next_state)
                    action_probs = mark_valid_moves(action_probs, valid_moves_flattened)
                    action_probs /= np.sum(action_probs)
                    node.expand(next_state, parent.to_play * -1, action_probs)

            # print("Action taken - ", action)
            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def new_root(self, root: Node):
        self.root = root

        self.root = root

    def backpropagate(self, search_path, value, to_play):
        """
        Backpropagates the evaluation value up the search path.

        Args:
            search_path (list): List of nodes in the search path.
            value (float): The evaluation value to propagate.
            to_play (int): The player to play at the root node.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1


if __name__ == "__main__":
    # Initialize hyperparameters, game, and model
    hyperparameters = Hyperparameters()
    game = OthelloGame()
    state = game.get_init_board()
    to_play = 1
    model = OthelloZeroModel(
        game.rows, game.get_action_size(), hyperparameters.Neural_Network["device"]
    )

    b = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, -1, 1, -1, -1, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # Run MCTS
    mcts = MCTS(game, model, hyperparameters)

    root = mcts.run(b, to_play)
    print(root)
