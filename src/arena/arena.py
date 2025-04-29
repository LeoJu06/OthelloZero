from multiprocessing import Pool
from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters
from src.mcts.mcts import MCTS
from src.othello.game_constants import PlayerColor
from src.utils.index_to_coordinates import index_to_coordinates

class Arena:

    def __init__(self):
        pass

    def play_single_game(self, challenger, old_model):
        """
        Simuliert ein einzelnes Spiel zwischen challenger und old_model.
        Gibt 1 zurück, wenn der Challenger gewinnt, -1 wenn er verliert.
        """
        game = OthelloGame()
        current_player = PlayerColor.BLACK.value

        mcts_challenger = MCTS(challenger)
        mcts_old_model = MCTS(old_model)

        state = game.get_init_board()
        turn = 0

        while not game.is_terminal_state(state):
            # Wähle MCTS basierend auf dem aktuellen Spieler
            mcts = mcts_challenger if current_player == PlayerColor.BLACK.value else mcts_old_model

            # MCTS-Suche für den aktuellen Spieler
            try:
                root = mcts.run_search(state, current_player)
            except AttributeError:
                print("Error Occured, Network outputed no valid moves")
                return 0

            # Aktion auswählen und Zustand aktualisieren
            x_pos, y_pos = index_to_coordinates(root.select_action(int(turn < 14)))
            state, current_player = game.get_next_state(state, current_player, x_pos, y_pos)

            root.reset()
            turn += 1

        # Gewinner bestimmen
        if game.determine_winner(state) == PlayerColor.BLACK.value:
            return 1  # Challenger gewinnt
        elif game.determine_winner(state) == PlayerColor.WHITE.value:
            return -1  # Challenger verliert
        return 0  # Unentschieden 

    def let_compete(self, challenger, old_model, num_games=None):
        """Lässt die zwei Modelle gegeneinander spielen und parallelisiert die Spiele."""
        num_games = num_games or Hyperparameters.Arena["arena_games"]
        num_workers = 5

        with Pool(num_workers) as pool:
            results = pool.starmap(self.play_single_game, [(challenger, old_model)] * num_games)

        # Ergebnisse auswerten
        won = results.count(1)
        lost = results.count(-1)

        print(f"Spiele abgeschlossen: {num_games}")
        print(f"Gewonnen: {won}")
        print(f"Verloren: {lost}")

        return won, lost
