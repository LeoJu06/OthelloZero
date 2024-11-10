
import constants as const
import src.logger_config as lg

class Board:
    def __init__(self, board, current_player=const.PlayerColor.BLACK):
        self.board = board
        self.current_player = current_player

    def apply_move(self):
        """Function to apply the given move into self.board"""
        
        self.switch_current_player()
        
    def switch_current_player(self):
        """Function to switch to the other player (e.g from black to white)"""
        self.current_player = -self.current_player

    def valid_moves(self):
        """Returns a list with all valid moves, which can be made"""
       
        valid_moves = []
        return valid_moves

    def is_draw(self):
        """Function to check wheter the board is a draw. """

        # Check whether the current player does not have a valid move
        if not self.valid_moves():
            # In this case, the current player would have to pass

            # Change to the opponents view
            self.switch_current_player()

            # Check whether the opponent no longer has a valid move
            if not self.valid_moves():
                # it is a draw, wether no one has a valid move or the board is full.
                return True  

            # change back to the original player
            # The logik for changing a player isn't implemented here.
            self.switch_current_player()
        
        # There is no draw yet
        return False  

    def print_board(self):
        """Function to print the board in a nice format to the terminal."""
        
        # Print the column numbers
        print("   0 1 2 3 4 5 6 7")
        print("  - - - - - - - - -")

        # Print the line number and the corresponding playing field values
        for row_idx, row in enumerate(self.board):
            
            row_str = f"{row_idx} |"  # Zeilenbezeichner, z.B. 0 | ...
            
            for col in row:
                if col == const.PlayerColor.BLACK.value:
                    row_str += " B"  # Schwarzer Stein
                elif col == const.PlayerColor.WHITE.value:
                    row_str += " W"  # Weißer Stein
                else:
                    row_str += " . "  # Leeres Feld (repräsentiert durch einen Punkt)
            
            # Drucke die formatierte Zeile
            print(row_str)


if __name__ == "__main__":

    field = [[-1] * 8 for x in range(8)]

    b = Board(field, 1)

    print(b.is_draw())

    b.print_board()
