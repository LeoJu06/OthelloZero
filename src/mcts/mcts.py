import time
import numpy as np
import src.utils.logger_config as lg
from src.config.hyperparameters import Hyperparameters
from src.mcts.node import Node
from src.neural_net.model import dummy_model_predict
from src.mcts.node import ucb_score
from src.othello.board import Board


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation with multiprocessing capabilities.
    """

    def __init__(self, root_node=None):
        """
        Initialize the MCTS object.

        Args:
            worker_id (int, optional): Unique identifier for the worker.
            root_node (Node, optional): The root node of the MCTS tree. If not provided, a new root node is created.
        """
        
        if root_node is None:
            self.root_node = Node()  # Initialize the root node
        else:
            self.root_node = root_node  # Use the provided root node.
       

    def get_best_move(self):
        """
        Method to return the best move with the corresponding child based on the highest number of visits.

        Returns:
            tuple: The best move (action) and the corresponding child node.
        """
        

        # Find the child with the highest visits using max with a key function
        best_move, best_child = max(
            self.root_node.children.items(), key=lambda item: item[1].visits
        )

        return best_move, best_child

    def search(self):

        start = time.time()


        action_probs, _ = dummy_model_predict()
        self.root_node.expand(action_probs)
        num_simulations = Hyperparameters.MCTS["num_simulations"]

        # Run the MCTS simulations
        for _ in range(num_simulations):
            node = self.root_node
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
        """
        # Simulation results: Print the results of the MCTS simulations
        print(f"Root's value => {self.root_node.value}")
        print(f"Root's visits => {self.root_node.visits}")
        print(
            f"Time needed for {num_simulations} iterations => {time.time() - start:.2f} seconds"
        )

        # Print details for each move
        for move, child in self.root_node.children.items():
            print(
                f"Move => {move}, Visits => {child.visits}, Value => {child.value:.2f}, UCB Score => {ucb_score(self.root_node, child):.2f}"
            )
        """



if __name__ == "__main__":

    


    
    mcts = MCTS()
    round = 0

    while not mcts.root_node.is_terminal_state():
        round += 1
        #print(f"Round={round}")
        last_player = mcts.root_node.board.player

        mcts.search()
        best_move, child = mcts.get_best_move()
        print(child.board.player)

        #print(f"Player {mcts.root_node.board.player} played")
        #print(f"Best Move found: {best_move}")
        
        mcts.root_node = child

        mcts.root_node.board.print_board()
       
        

        




    