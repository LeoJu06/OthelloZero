
from src.othello.othello_game import OthelloGame
from src.othello.game_constants import PlayerColor
from src.mcts.node import Node
class DataManager:
    

    def __init__(self):
        self.data  =[]
        self.game = OthelloGame()

    def create_example(self, state, current_player, root:Node):
        if current_player == PlayerColor.BLACK.value:
            state = self.game.get_canonical_board(state, current_player):

        pi = None


    def collect(self, x):

        self.data.append(x)