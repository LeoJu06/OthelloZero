import graphviz
from src.othello.othello_game import OthelloGame
from src.mcts.mcts import MCTS
from src.neural_net.model import OthelloZeroModel
from src.config.hyperparameters import Hyperparameters


def visualize_game_tree(root, max_depth=5):
    """
    Visualizes the game tree starting from the root node using Graphviz.

    Args:
        root (Node): The root node of the game tree.
        max_depth (int): The maximum depth of the tree to visualize.

    Returns:
        graphviz.Digraph: The Graphviz graph object for the game tree.
    """

    def add_node(graph, node, label, depth, node_id):
        """
        Recursively adds nodes and edges to the graph.

        Args:
            graph (graphviz.Digraph): The Graphviz graph object.
            node (Node): The current node.
            label (str): Label for the current node.
            depth (int): Current depth in the tree.
            node_id (str): Unique identifier for the node.
        """
        if depth > max_depth:
            return

        # Add the current node
        graph.node(node_id, label=label)

        # Add children
        for action, child in node.children.items():
            child_id = f"{node_id}_{action}"
            action_label = f"Action: {action}\nPrior: {child.prior:.2f}\nVisits: {child.visit_count}\nValue: {child.value():.2f}"
            graph.node(child_id, label=action_label)
            graph.edge(node_id, child_id)

            # Recursively add child nodes
            add_node(graph, child, action_label, depth + 1, child_id)

    # Create a directed graph
    graph = graphviz.Digraph(format="png")
    graph.attr(rankdir="TB")  # Top to Bottom direction

    # Add the root node
    root_label = f"Root\nVisits: {root.visit_count}\nValue: {root.value():.2f}"
    add_node(graph, root, root_label, 0, "root")

    return graph


# Example usage
if __name__ == "__main__":
    import numpy as np

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
    # Run MCTS to generate the tree
    hyperparameters = Hyperparameters()
    game = OthelloGame()
    state = game.get_init_board()
    model = OthelloZeroModel(
        game.rows, game.get_action_size(), hyperparameters.Neural_Network["device"]
    )
    mcts = MCTS(game, model, hyperparameters)
    root = mcts.run(b, -1)

    # Visualize the tree
    tree_graph = visualize_game_tree(root, max_depth=3)
    tree_graph.render("game_tree", view=True)  # Saves and opens the tree visualization
