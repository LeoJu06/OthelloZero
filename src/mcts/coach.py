from src.config.hyperparameters import Hyperparameters
from src.othello.othello_game import OthelloGame
from src.mcts.mcts import MultiprocessedMCTS, MCTS
from src.neural_net.model import OthelloZeroModel
from src.data_manager.data_manager import DataManager
from src.mcts.manager import init_manager, init_manager_process,terminate_manager_process, create_multiprocessed_mcts
from src.othello.game_constants import PlayerColor
from src.utils.index_to_coordinates import index_to_coordinates
import torch.multiprocessing as mp
from tqdm import tqdm
import time


class Coach:
    """
    The Coach class manages self-play episodes, data collection, and training 
    for the Othello Zero model.
    """

    def __init__(self):
        """
        Initializes the Coach with hyperparameters and a data manager.
        """
        self.hyperparams = Hyperparameters()
        self.data_manager = DataManager()

    def execute_single_episode(self, mcts: MultiprocessedMCTS):
        """
        Runs a single self-play episode using MCTS.

        Args:
            mcts (MultiprocessedMCTS): The Monte Carlo Tree Search instance.

        Returns:
            list: Training examples generated during the episode.
        """
        game = OthelloGame()
        state = game.get_init_board()
        current_player = PlayerColor.BLACK.value  # Black always starts
        examples = []  # Training data storage
        episode_step = 0
        local_data_manager = DataManager()

        while not game.is_terminal_state(state):
           # start  = time.time()
            temp = int(episode_step < self.hyperparams.MCTS["temp_threshold"])
            root = mcts.run_search(state, current_player)

            # Store training example
            examples.append(local_data_manager.create_example(state, current_player, root, temp))

            # Select action and update game state
            x_pos, y_pos = index_to_coordinates(root.select_action(temp))
            state, current_player = game.get_next_state(state, current_player, x_pos, y_pos)

            episode_step += 1
            root.reset()
            end = time.time()
            #print(f"Time per move {end-start:.2f}")

        # Assign rewards based on the game outcome
        game_outcome = game.get_reward_for_player(state, PlayerColor.BLACK.value)
        return local_data_manager.assign_rewards(examples, game_outcome)

    def self_play(self, multi_mcts: MultiprocessedMCTS):
        """
        Runs multiple self-play episodes and collects training examples.

        Args:
            multi_mcts (MultiprocessedMCTS): The MCTS instance.
            num_episodes (int): Number of episodes to run.
        """
        all_examples = []
        iters = self.hyperparams.Coach["episodes_per_worker"]
        for _ in tqdm(range(iters)):
            print(f"Episode {_}/{iters}, Worker  {multi_mcts.idx}")
            all_examples += self.execute_single_episode(multi_mcts)

        # TODO: Implement data saving mechanism
        self.data_manager.save_training_examples(all_examples)

    def learn(self):
        """
        Runs the reinforcement learning loop, collecting self-play data and training the model.
        """

        g = OthelloGame()
        h = self.hyperparams
        model = OthelloZeroModel(g.rows, g.get_action_size(), h.Neural_Network["device"])
        for iteration in range(1, self.hyperparams.Coach["iterations"] + 1):
            start = time.time()
            print(f"Iteration {iteration}/{self.hyperparams.Coach['iterations']} - Starting self-play...")

            #model = self.data_manager.load_best_model()
            # Initialize MCTS with multiprocessing
            manager = init_manager(model, self.hyperparams)
            manager_process = init_manager_process(manager)

           
            workers = []
            for worker_id in range(self.hyperparams.Coach["num_workers"]):
                multi_mcts = create_multiprocessed_mcts(worker_id, manager)
                worker_process = mp.Process(target=self.self_play, args=(multi_mcts,))
                workers.append(worker_process)
                worker_process.start()
            for worker in workers:
                worker.join()

            terminate_manager_process(manager_process)

            print(f"Iteration {iteration} - Self-play complete. Training model...")

            end = time.time()
            print(f"Iter needed {end-start:.2f}s")

            self.train()

    def train(self):
        """
        Trains the neural network using collected self-play data.
        """
        # TODO: Implement training mechanism
        pass


if __name__ == "__main__":
    """
    Main execution block to initialize the game, model, and coach, and run a self-play episode.
    """
    coach =  Coach()
    coach.learn()