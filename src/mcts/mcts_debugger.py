from graphviz import Digraph
import numpy as np
from mcts import MCTS
from node import Node
from src.othello.board import Board


class MCTSDebugger:
    def __init__(self, root_node, max_depth=4):
        self.root_node = root_node
        self.max_depth = max_depth

    def log_tree(self, node=None, depth=0):
        """
        Logs the tree structure to the console, showing node values and visits.

        Args:
            node (Node): Current node to log.
            depth (int): Depth of the node in the tree.
        """
        if node is None:
            node = self.root_node

        indent = "  " * depth
        print(
            f"{indent}Move: {getattr(node.board, 'last_move', None)}, Visits: {node.visits}, \
              Value: {node.value:.2f}, Prior: {node.prior:.2f}"
        )

        for child in node.children.values():
            self.log_tree(child, depth + 1)

    def visualize_tree(self, node=None, graph=None, depth=0):
        """
        Visualizes the MCTS tree using Graphviz.

        Args:
            node (Node): Current node to visualize.
            graph (Digraph): The Graphviz Digraph object.
            depth (int): Current depth of the visualization.

        Returns:
            graph: The Graphviz Digraph object.
        """
        if node is None:
            node = self.root_node
        if graph is None:
            graph = Digraph(format="png")
            graph.attr(dpi="300")

        # Convert the board to a string representation for the label
        board_str = node.board.print_board(False)
        
        node_label = (
            f"Move: {getattr(node.board, 'last_move', None)}\n"
            f"Player: {node.board.player}\n"
            f"Visits: {node.visits}\n"
            f"Value: {node.value:.2f}\n"
            f"Prior: {node.prior:.2f}\n"
            f"Board:\n{board_str}"
        )
        graph.node(str(id(node)), label=node_label, shape="box")

        if depth < self.max_depth:
            for action, child in node.children.items():
                graph.edge(
                    str(id(node)),
                    str(id(child)),
                    label=f"{action} (player={node.board.player})",
                )
                self.visualize_tree(child, graph, depth + 1)

        return graph


    def inspect_predictions(self, node, model):
        """
        Logs neural network predictions for a given node.

        Args:
            node (Node): The node to evaluate.
            model: Neural network model to use for predictions.
        """
        action_probs, value = model.predict(node)
        print(f"Predicted Action Probs: {action_probs}, Predicted Value: {value:.2f}")

    def interactive_debug(self, mcts):
        """
        Allows step-by-step interactive debugging of the MCTS.

        Args:
            mcts (MCTS): The MCTS object to debug.
        """
        while not mcts.root_node.is_terminal_state():
            print(f"Current Player: {mcts.root_node.board.player}")
            print(f"Board State:\n{mcts.root_node.board}")

            print("Search Starting...")
            mcts.search()
            print("Search Completed.")

            print("MCTS Tree:")
            self.log_tree()

            best_move, child = mcts.get_best_move()
            print(f"Best Move: {best_move}, Child Value: {child.value:.2f}")

            graph = self.visualize_tree()
            graph.render("mcts_debug_tree")
            print("Tree visualization saved as 'mcts_debug_tree.png'")

            input("Press Enter to continue...")
            mcts.root_node = child


# Example Usage:
# Assuming you have MCTS and Node objects implemented, you can use this debugger as follows:
#

#

if __name__ == "__main__":
    board = [[-1] * 8 for x in range(8)]
    board[0][1] = 0
    board[0][2] = 0
    board[0][4] = 1
    board[1][1] = 1

    board = [
    [0, -1, -1, -1, -1,  1,  1,  1],
    [1,  1,  1,  1, -1, -1, -1, -1],
    [1,  1,  1,  1, -1,  1, -1, -1],
    [1,  1,  1, -1,  1,  1,  1,  1],
    [1,  1,  1,  1, -1, -1,  1,  1],
    [1,  1,  1,  1, -1,  1,  1, -1],
    [1,  1,  1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1, -1,  1,  0, -1],
]
    




    b = Board(board=board, player=-1)
    #b.apply_move(7, 6)
    #b.update()
    #b.apply_move(0, 0)
    #b.update()
    #print(b.determine_winner())
    
    
    b.print_board()

    root_node = Node(board=b)
    mcts = MCTS(root_node=root_node)
    #
    debugger = MCTSDebugger(root_node)
    debugger.interactive_debug(mcts)
