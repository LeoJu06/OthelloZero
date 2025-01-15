import numpy as np
import math


def ucb_score(parent: "Node", child: "Node") -> float:
    """
    Calculates the Upper Confidence Bound (UCB) score for a given child node.

    Args:
        parent (Node): The parent node in the tree.
        child (Node): The child node for which the UCB score is calculated.

    Returns:
        float: The computed UCB score.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    value_score = -child.value() if child.visit_count > 0 else 0
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

    def __init__(self, prior: float, to_play: int):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self) -> bool:
        """
        Checks if the node has been expanded (i.e., has children).

        Returns:
            bool: True if the node has children, False otherwise.
        """
        return len(self.children) > 0

    def value(self) -> float:
        """
        Returns the average value of the node.

        Returns:
            float: Average value, or 0 if unvisited.
        """
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def select_action(self, temperature: float) -> int:
        """
        Selects an action based on the visit count distribution and a temperature parameter.

        Args:
            temperature (float): Determines the level of exploration (0 for greedy, inf for random).

        Returns:
            int: The selected action.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())

        if temperature == 0:
            return actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            return np.random.choice(actions)
        else:
            # Adjust distribution based on temperature
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution /= visit_count_distribution.sum()
            return np.random.choice(actions, p=visit_count_distribution)

    def select_child(self) -> tuple:
        """
        Selects the child with the highest UCB score.

        Returns:
            tuple: The best action and the corresponding child node.
        """
        best_score = -np.inf
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state: np.ndarray, to_play: int, action_probs: np.ndarray):
        """
        Expands the node by creating child nodes for each valid action.

        Args:
            state (np.ndarray): The board state at this node.
            to_play (int): The player to play at this node.
            action_probs (np.ndarray): The prior probabilities for each action.
        """
        self.to_play = to_play
        self.state = state

        if action_probs[-1] != 0:
            # No valid moves, so the player must pass
            self.children[-1] = Node(prior=action_probs[-1], to_play=-to_play)
            return

        for move, prob in enumerate(action_probs):
            if prob > 0:
                self.children[move] = Node(prior=prob, to_play=-to_play)

    def __repr__(self):
        """
        Provides a string representation of the node for debugging purposes.

        Returns:
            str: A formatted string representing the node.
        """
        return f"State: {self.state} Prior: {self.prior:.2f} Count: {self.visit_count} Value: {self.value()[0]}"
