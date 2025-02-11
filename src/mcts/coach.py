from src.config.hyperparameters import Hyperparameters
from src.othello.othello_game import OthelloGame
from src.mcts.mcts import MultiprocessedMCTS, MCTS
from src.neural_net.model import OthelloZeroModel
from src.data_manager.data_manager import DataManager
from src.mcts.manager import Manager
from src.othello.game_constants import PlayerColor
from src.utils.index_to_coordinates import index_to_coordinates

class Coach:
    """
    The Coach class manages self-play episodes, data collection, and training for the Othello Zero model.
    """
    def __init__(self):
        """
        Initializes the Coach with hyperparameters and a data manager.
        """
        self.hyperparams = Hyperparameters()
        self.data_manager = DataManager()

    def execute_episode(self, mcts: MultiprocessedMCTS):
        """
        Executes a single self-play episode using MCTS.

        Args:
            mcts (MultiprocessedMCTS): The Monte Carlo Tree Search instance.

        Returns:
            list: A list of training examples generated during the episode.
        """
        current_player = PlayerColor.BLACK.value  # Black starts first
        examples = []  # Stores training examples
        episode_step = 0  # Tracks the number of moves in the episode
        game = OthelloGame()
        state = game.get_init_board()
        local_data_manager = DataManager()
        
        # Play until reaching a terminal state
        while not game.is_terminal_state(state):
            # Temperature parameter for exploration
            temp = int(episode_step < self.hyperparams.MCTS["temp_threshold"])
            
            # Run MCTS to get the root node with action probabilities
            root = mcts.run_search(state, current_player)
            
            # Create and store an example
            example = local_data_manager.create_example(state, current_player, root, temp)
            examples.append(example)

            # Select an action and get the next state
            x_pos, y_pos = index_to_coordinates(root.select_action(temp))
            state, current_player = game.get_next_state(state, current_player, x_pos, y_pos)
            
            episode_step += 1
            root.reset()  # Reset MCTS root for the next iteration

        # Assign rewards based on game outcome
        game_outcome = OthelloGame().get_reward_for_player(state, PlayerColor.BLACK.value)
        examples = local_data_manager.assign_rewards(examples, game_outcome)

        return examples

    def policy_iteration_self_play(self):
        """
        Placeholder method for policy iteration through self-play.
        """
        pass

    def learn(self):
        """
        Executes the learning loop, collecting self-play data and training the model.
        """
        train_examples = []

        # TODO: Define the number of iterations
        for i in range(1, self.hyperparams):
            
            # TODO: Define the number of epochs per iteration
            for epoch in range(self.hyperparams): 
                examples = self.execute_episode(mcts=None)  # TODO: Provide a valid MCTS instance
                self.data_manager.collect(examples)
            
            # TODO: Implement data saving mechanism
            self.train()

    def train(self):
        """
        Placeholder method for training the neural network with collected data.
        """
        pass

if __name__ == "__main__":
    """
    Main execution block to initialize the game, model, and coach, and run a self-play episode.
    """
    g = OthelloGame()
    h = Hyperparameters()
    coach = Coach()
    
    # Initialize the model with game parameters
    model = OthelloZeroModel(g.rows, g.get_action_size(), h.Neural_Network["device"])
    mcts = MCTS(model)
    
    # Run a single episode and print the results
    examples = coach.execute_episode(mcts)
    print(examples)
