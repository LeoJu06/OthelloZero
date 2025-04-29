from src.othello.game_constants import PlayerColor
from src.othello.othello_game import OthelloGame
from src.neural_net.model import OthelloZeroModel
from src.mcts.node import Node
import json 
import os
from src.utils.no_duplicates import filter_duplicates
import pickle
import pickle
import torch

class DataManager:
    def __init__(self):
        """Initializes the DataManager."""
        self.data = []
        self.game = OthelloGame()

    def create_example(self, current_state, player, root: Node, temperature):
        """Creates a training example.

        Args:
            current_state: The current game state.
            player: The current player.
            root: The root node of the MCTS tree.
            temperature: The temperature for policy calculation.

        Returns:
            A list containing the canonical board, policy, and reward (initially None).
        """
        if player == PlayerColor.BLACK.value:
            current_state = self.game.get_canonical_board(current_state, player)
        return [current_state, root.pi(temperature), None]

    def assign_rewards(self, examples, game_outcome):
        """Assigns rewards to the examples based on the game outcome.

        Args:
            examples: A list of training examples.
            game_outcome: The outcome of the game (1 for win, -1 for loss).

        Returns:
            The updated list of examples with rewards assigned.
        """
        for example in examples:
            example[2] = game_outcome
            game_outcome *= -1  # Alternate rewards for players
        return examples

    def _path_to_data_dir(self):
        """Returns the path to the data directory."""
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

    def get_iter_number(self):
        """Returns the current iteration number.

        Returns:
            The current iteration number as an integer.
        """
        data_dir = self._path_to_data_dir()
        path = os.path.join(data_dir, "iteration_number.txt")

        # Ensure the file exists and is initialized
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("0")  # Initialize with 0 if the file doesn't exist

        with open(path, "r") as f:
            n = f.read().strip()

        # Validate the content of the file
        if not n.isdigit():
            raise ValueError(f"Invalid content in {path}. Expected a number, got '{n}'.")

        return int(n)

    def increment_iteration(self):
        """Increments the iteration number by 1."""
        n = self.get_iter_number()
        data_dir = self._path_to_data_dir()
        path = os.path.join(data_dir, "iteration_number.txt")

        with open(path, "w") as f:
            f.write(str(n + 1))  # Write the incremented value

    def save_training_examples(self, examples):
        """Saves training examples to a file.

        Args:
            examples: A list of training examples to save.
        """
        data_dir = self._path_to_data_dir()
        n = self.get_iter_number()
        filename = f"examples/examples_iteration_{n}.pkl"
        path = os.path.join(data_dir, filename)

        with open(path, "wb") as f:
            pickle.dump(examples, f)


    def load_examples(self, n=None):
        """
        Load training examples from disk, either from a specific iteration or multiple recent iterations.
        
        Args:
            n: Optional[int]. If specified, loads examples from this specific iteration number.
                If None, loads examples from the last 5 iterations (excluding current).
                
        Returns:
            List of training examples. When loading multiple iterations, duplicates are removed.
            
        Raises:
            FileNotFoundError: If specified iteration file doesn't exist.
            pickle.UnpicklingError: If a pickle file is corrupted.
        """
        data_dir = self._path_to_data_dir()

        # Case 1: Load specific iteration
        if n is not None:
            filename = f"examples/examples_iteration_{n}.pkl"
            path = os.path.join(data_dir, filename)
            
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Training examples file not found for iteration {n}")
            except pickle.UnpicklingError as e:
                raise pickle.UnpicklingError(f"Corrupted pickle file for iteration {n}: {str(e)}")

        # Case 2: Load multiple recent iterations (with deduplication)
        current_iter = self.get_iter_number()
        combined_examples = []
        
        # Load examples from last 5 completed iterations (n-5 to n-1)
        start_iter = max(current_iter - 9, 0)  # Ensure we don't go negative
        end_iter = max(current_iter, 1)  # Ensure at least 1 iteration exists
        
        for iter_num in range(start_iter, end_iter):
            print(iter_num)
            try:
                filename = f"examples/examples_iteration_{iter_num}.pkl"
                path = os.path.join(data_dir, filename)
                
                with open(path, "rb") as f:
                    examples = pickle.load(f)
                    combined_examples.extend(examples)
                    
            except FileNotFoundError:
                print(f"Warning: Examples file missing for iteration {iter_num}")
                raise FileNotFoundError
                continue
            except pickle.UnpicklingError:
                print(f"Warning: Corrupted examples file for iteration {iter_num}")
                continue

       
        return combined_examples

        


    
    def save_model(self, model:OthelloZeroModel):
        n = self.get_iter_number()
        torch.save(model.state_dict(), f"data/models/othello_zero_model_{n}")

    def load_model(self, latest_model=True, n=None):
        """Loads a model. If n is None the best model (last iter) is being returned"""

        if latest_model:
            n = self.get_iter_number()
        model = OthelloZeroModel(board_size=8, action_size=65, device="cuda")
        model.load_state_dict(torch.load(f"data/models/othello_zero_model_{n}"))
        return model


    def collect(self, training_example):
        """Collects training examples.

        Args:
            training_example: A single training example to add to the data list.
        """
        self.data.append(training_example)


    def save_report_as_json(self, won, lost):
        """Creates a report.json file. Where all training information is stored"""

        pass 
        raise NotImplementedError







if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    print(f"Data directory: {data_dir}")

    da = DataManager()
  
    n = da.get_iter_number()

    e = da.load_examples()
    print(len(e))
    e = filter_duplicates(e)
    print(e[0])
    print(len(e[0][1]))

