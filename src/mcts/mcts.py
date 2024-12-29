"""MCTS File which contains the MCTS class."""
import src.utils.logger_config as lg
from src.config.hyperparameters import Hyperparameters
from src.mcts.node import Node
from src.neural_net.model import dummy_model_predict



class MCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation for singel core usage.
    How To Use: 
        Simply create a MCTS object. Pass a starting node, otherwise the mcts assumes to begin 
        from an empty board aka the father of root nodes:).
    """

    def __init__(self, root_node=None):
        """
        Initialize the MCTS object.

        Args:
            root_node (Node, optional): The root node of the MCTS tree. If not provided, a new root node is created.
        """
        self.root_node = root_node if root_node else Node()

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
       
        # TODO connect mcts with neural network
        action_probs, _ = dummy_model_predict()
        self.root_node.expand(action_probs)
        num_simulations = Hyperparameters.MCTS["num_simulations"]

        # Run the MCTS simulations
        for _ in range(num_simulations):
            print(f"Simulation {_}/{num_simulations}", end="\r")
            node = self.root_node
            search_path = [node]  # Keep track of the nodes visited in this simulation

            # Selection phase: Traverse the tree to select a node to expand
            while node.is_expanded():
                (
                    action,
                    node,
                ) = (
                    node.select_child()
                )  # Select the child node with the highest UCB score
                search_path.append(node)

            # Evaluation phase: Evaluate the value of the node (either terminal or predicted)
            value = None
            if node.is_terminal_state():
                value = (
                    node.board.determine_winner()
                )  # If terminal, use the winner value
            if value is None:
                action_probs, value = dummy_model_predict(
                    node.board
                )  # Predict value using the model if not terminal
                node.expand(
                    action_probs
                )  # Expand the node with the new action probabilities
            

            # Backpropagation phase: Update the value and visit count of the nodes along the search path
            # TODO checkout if -value has to be backpropagated.
            for node in search_path:
                node.value += value  # Update node value based on simulation outcome
                node.visits += 1  # Increment the number of visits to this node
     
    def show_results(self):

        print(f"Root Node expanded {self.root_node.visits} times")
        
        print(f"Root's value => {self.root_node.value}")
        print("Stats of childs: ")
        for move, child in self.root_node.children.items():

            print(f"    Child {move}, visits=>{child.visits}, value=>{child.value}")
        
       
        

if __name__ == "__main__":
    mcts = MCTS()
    

    while not mcts.root_node.is_terminal_state():
        mcts.search()
        mcts.show_results()

        best_move, child = mcts.get_best_move()
        mcts.root_node = child
        mcts.root_node.board.print_board()
