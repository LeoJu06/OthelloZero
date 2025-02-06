from src.config.hyperparameters import Hyperparameters
from src.othello.othello_game import OthelloGame
from src.mcts.mcts import MultiprocessedMCTS
from src.data_manager.data_manager import DataManager
from src.mcts.manager import Manager
from src.othello.game_constants import PlayerColor
from src.utils.index_to_coordinates import index_to_coordinates

class Coach:


    def __init__(self):
        self.hyperparams = Hyperparameters()
        self.data_manager = DataManager()

    def execute_episode(self, mcts:MultiprocessedMCTS):
        
       
        current_player = PlayerColor.BLACK.value
        examples = []
        episode_step = 0
        game = OthelloGame()
        state = game.get_init_board()
        

        while not game.is_terminal_state(state):
            temp = int(episode_step < self.hyperparams.MCTS["temp_threshold"])

            root = mcts.run_search(state, current_player)

            example = self.data_manager.create_example(state, current_player, root)
            examples.append(example)


         

            action_x, action_y = index_to_coordinates(root.select_action(temp))

            episode_step += 1
            root.reset()

        return examples

    def policy_iteration_self_play(self):
        pass


    def learn(self):

        train_examples = []

        #TODO add iter number
        for i in range(1, self.hyperparams):
            
            for epoch in range(self.hyperparams): #TODO Add num epoch

                examples = self.execute_episode(mcts=None)
                self.data_manager.collect(examples)

            # TODO Add data saving


            self.train()

        


        




    def train(self):
        pass