


def from_index_to_algebraic(index: int) -> str:
    row = index // 8
    col = index % 8
    return f"{chr(col + ord('a'))}{8 - row}"  # beachte: 8 - row