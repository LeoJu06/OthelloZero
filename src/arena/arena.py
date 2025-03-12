from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters
from src.mcts.mcts import MCTS
from src.othello.game_constants import PlayerColor
from src.utils.index_to_coordinates import index_to_coordinates

class Arena:


    def __init__(self):
        pass

    def let_compete(self, challenger, old_model):
        """Performs self plays of 2 models. 
        Returns (Won, Lost)"""
        game = OthelloGame()
        current_player = PlayerColor.BLACK.value

        mcts_challenger = MCTS(challenger)
        mcts_old_model = MCTS(old_model)

        won = 0
        lost = 0

        for game_to_play in range(Hyperparameters.Arena["arena_games"]):
            state = game.get_init_board()


            while not game.is_terminal_state(state):
                # Alternate between challenger and old model MCTS based on the current player
                if current_player == PlayerColor.BLACK.value:
                    mcts = mcts_challenger
                else:
                    mcts = mcts_old_model

                # Run MCTS search for the current player
                root = mcts.run_search(state, current_player)

                # Select action based on MCTS output
                x_pos, y_pos = index_to_coordinates(root.select_action(1))

                # Get the next state and switch players
                state, current_player = game.get_next_state(state, current_player, x_pos, y_pos)

                # Reset MCTS for the next iteration
                root.reset()

            if game.determine_winner(state) == PlayerColor.BLACK.value:
                won += 1
            elif game.determine_winner(state) == PlayerColor.WHITE.value:
                lost += 1

        print("win", won)
        print("lost", lost)

        

        return won, lost
    
    


            
            








        