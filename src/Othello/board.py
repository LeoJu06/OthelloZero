"""This File contains the board class, 
which represents an othello board with all it's functions
"""
import constants as const
import src.logger_config as lg

class Board:
    """The Board Class represents an Othello board
    """
    def __init__(self, board, current_player=const.PlayerColor.BLACK):
        self.board = board
        self.current_player = current_player

    def apply_move(self, x_pos: int, y_pos: int):
        """Function to apply the given move into self.board.
        Assumes that only valid moves are passed to this method.
        """
        self.board[x_pos][y_pos] = self.current_player
        lg.logger_board.debug("Player (%s) places at (%s|%s)", self.current_player, x_pos, y_pos)
        self.switch_current_player()

    def switch_current_player(self):
        """Function to switch to the other player (e.g from black to white)"""
        self.current_player = -self.current_player

    def _is_on_board(self, x_pos, y_pos):

        """Method to determine wheter the field with the given coordinates is on the board"""
        return (0 <= x_pos < 8) and (0 <= y_pos < 8)

    def valid_moves(self):
        """Returns a list with all valid moves, which can be made"""
        valid_moves = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # Gehe durch alle Felder
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == const.EMPTY:  # Nur leere Felder
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        found_opponent = False

                        # Gehe weiter in der Richtung und suche nach gegnerischen Steinen
                        while 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == -self.current_player:
                            found_opponent = True
                            nx += dx
                            ny += dy

                        # Wenn ein gegnerischer Stein gefunden wurde, prüfe, ob der Zug gültig ist
                        if found_opponent and 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == self.current_player:
                            valid_moves.append((x, y))
                            break  # Wenn ein gültiger Zug gefunden wurde, breche die Schleife ab

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

def test_speed_of_valid_moves():

    import time

    start = time.time()
    iter = 10**2

    for i in range(iter):
        b.valid_moves()
    
    t = time.time() -start

    print(f"Time needed to run for {iter} iterations: {t} seconds")



if __name__ == "__main__":

    field = [[const.EMPTY] * 8 for x in range(8)]

    b = Board(field, const.PlayerColor.BLACK.value)
    b.apply_move(3, 4)
    b.apply_move(4, 4)
    b.apply_move(4, 3)
    b.apply_move(3, 3)


    print(b.is_draw())

    b.print_board()
    print(b.valid_moves())


    test_speed_of_valid_moves()
