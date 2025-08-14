

def from_algebraic_to_coords(move: str) -> tuple[int, int]:
    col = ord(move[0]) - ord('a')
    row = 8 - int(move[1])  # wichtig: 8 - ... → da Pygame von oben zählt
    return row, col