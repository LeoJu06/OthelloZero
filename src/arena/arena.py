from multiprocessing import Pool
from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters, mcts_simulations
from src.mcts.mcts import MCTS
from src.othello.game_constants import PlayerColor
from src.utils.index_to_coordinates import index_to_coordinates
from src.data_manager.data_manager import DataManager
import numpy as np
import random


class Arena:

    def __init__(self):
        self.game = OthelloGame()

    def play_single_game(self, challenger, old_model):
        game = OthelloGame()
        current_player = PlayerColor.BLACK.value

        mcts_challenger = MCTS(challenger)
        mcts_old_model = MCTS(old_model)

        state = game.get_init_board()
        turn = 0
        num_simulations = mcts_simulations(DataManager().get_iter_number())

        while not game.is_terminal_state(state):
            mcts = mcts_challenger if current_player == PlayerColor.BLACK.value else mcts_old_model

            try:
                root = mcts.run_search(state, current_player, num_simulations=num_simulations)
            except AttributeError:
                print("Error Occurred, Network output no valid moves")
                return 0

            x_pos, y_pos = index_to_coordinates(root.select_action(int(turn < 14)))
            state, current_player = game.get_next_state(state, current_player, x_pos, y_pos)

            root.reset()
            turn += 1

        winner = game.determine_winner(state)
        if winner == PlayerColor.BLACK.value:
            return 1
        elif winner == PlayerColor.WHITE.value:
            return -1
        return 0

    def let_compete(self, challenger, old_model, num_games=None):
        num_games = num_games or Hyperparameters.Arena["arena_games"]
        num_workers = 5

        with Pool(num_workers) as pool:
            results = pool.starmap(self.play_single_game, [(challenger, old_model)] * num_games)

        won = results.count(1)
        lost = results.count(-1)

        print(f"Spiele abgeschlossen: {num_games}")
        print(f"Gewonnen: {won}")
        print(f"Verloren: {lost}")

        return won, lost

    def model_vs_minimax(self, model, num_games=10, depth=3):
        game = self.game
        mcts_model = MCTS(model)

        results = []

        for i in range(num_games):
            state = game.get_init_board()
            current_player = PlayerColor.BLACK.value if i % 2 == 0 else PlayerColor.WHITE.value
            model_color = current_player
            turn = 0

            while not game.is_terminal_state(state):
                if current_player == model_color:
                    root = mcts_model.run_search(state, current_player, num_simulations=1000)
                    move = index_to_coordinates(root.select_action(int(turn < 14)))
                    root.reset()
                else:
                    move = self.minimax_move(game, state, current_player, depth)

                if move is None:
                    break

                state, current_player = game.get_next_state(state, current_player, *move)
                turn += 1

            winner = game.determine_winner(state)
            if winner == model_color:
                results.append(1)
            elif winner == -model_color:
                results.append(-1)
            else:
                results.append(0)

        won = results.count(1)
        lost = results.count(-1)
        print(f"Modell gegen Minimax (Tiefe {depth}): Gewonnen: {won}, Verloren: {lost}, Unentschieden: {results.count(0)}")
        return won, lost

    def minimax_move(self, game, state, player_color, depth=3):
        best_score = float('-inf')
        best_move = None

        for move in game.get_valid_moves(state, player_color):
            next_state, _ = game.get_next_state(state, player_color, *move)
            score = self.minimax(game, next_state, depth - 1, False, player_color, float('-inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def minimax(self, game, state, depth, maximizing_player, player_color, alpha, beta):
        if depth == 0 or game.is_terminal_state(state):
            return self.evaluate_board(state, player_color)

        current_color = player_color if maximizing_player else -player_color
        valid_moves = game.get_valid_moves(state, current_color)

        if not valid_moves:
            return self.minimax(game, state, depth - 1, not maximizing_player, player_color, alpha, beta)

        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                next_state, _ = game.get_next_state(state, current_color, *move)
                eval = self.minimax(game, next_state, depth - 1, False, player_color, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                next_state, _ = game.get_next_state(state, current_color, *move)
                eval = self.minimax(game, next_state, depth - 1, True, player_color, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_board(self, state, player_color):
        opponent = -player_color

        # Ecken kontrollieren
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        corner_score = sum([
            10 if state[x, y] == player_color else -10 if state[x, y] == opponent else 0
            for x, y in corners
        ])

        # Mobilität
        my_moves = len(self.game.get_valid_moves(state, player_color))
        opp_moves = len(self.game.get_valid_moves(state, opponent))
        mobility_score = 0
        if my_moves + opp_moves != 0:
            mobility_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

        # Steinvorteil (nicht zu hoch gewichten!)
        disc_diff = (state == player_color).sum() - (state == opponent).sum()

        return 5 * corner_score + 2 * mobility_score + 1 * disc_diff
    
    def count_policy_from_fixed_state(self, model, move_counts, num_simulations=1000, target_turn=8):
        """
        Führt ein Spiel bis zu einem bestimmten Zug aus und sammelt dort die normalisierte MCTS-Policy.
        Ziel: Heatmap der Modellpolicy (wie bei OLIVAW)
        """
        game = OthelloGame()
        state = game.get_init_board()
        current_player = PlayerColor.BLACK.value
        turn = 0

        # Zufällige Spielerfarbe für das Modell (nur wenn du später noch duellierst, hier egal)
        model_color = random.choice([PlayerColor.BLACK.value, PlayerColor.WHITE.value])

        while not game.is_terminal_state(state) and turn < target_turn:
            root = MCTS(model).run_search(state, current_player, num_simulations=num_simulations)
            action = root.select_action(temperature=1 if turn < 8 else 0)
            if action == -1:
                break
            x, y = index_to_coordinates(action)
            state, current_player = game.get_next_state(state, current_player, x, y)
            turn += 1

        # Jetzt bei gewünschtem Zug MCTS laufen lassen und Policy extrahieren
        if not game.is_terminal_state(state):
            root = MCTS(model).run_search(state, current_player, num_simulations=num_simulations)

            # Extrahiere und normiere Visit-Counts
            total_visits = sum(c.visit_count for a, c in root.children.items() if a != -1)
            if total_visits > 0:
                for action, child in root.children.items():
                    if action != -1:
                        row, col = index_to_coordinates(action)
                        move_counts[row, col] += child.visit_count / total_visits


    def count_moves_over_games(self, model, num_games=100, num_simulations=1000):
        move_counts = np.zeros((8, 8), dtype=np.int32)
        for i in range(num_games):
            self.count_policy_from_fixed_state(model, move_counts, num_simulations)
        return move_counts




if __name__ == "__main__":
    from src.data_manager.data_manager import DataManager

    arena = Arena()
    dm = DataManager()

    model = dm.load_model()

    arena.model_vs_minimax(model, depth=4)
