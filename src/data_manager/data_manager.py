from src.othello.game_constants import PlayerColor
from src.othello.othello_game import OthelloGame
import numpy as np
from src.mcts.node import Node

class DataManager:

    def __init__(self):

        self.data = []
        self.game = OthelloGame()
        
        pass

    def create_example(self, current_state, player, root:Node, temperature):
        """Creates an example"""
        
        if player == PlayerColor.BLACK.value:
            current_state = self.game.get_canonical_board(current_state, player)

        return [current_state, root.pi(temperature), None]
    
    def assign_rewards(self, examples, game_outcome):

        for situation in examples:

            situation[2] = game_outcome
            game_outcome *= -1

        return examples
    
    def increment_iteration(self):

        pass

    def save_training_examples(self, examples):
        pass

    def load_best_model(self):
        best_model = None
        return best_model

        
        




    def collect(self, training_example):
        """Collects data examples"""

        self.data.append(training_example)