def ContainsSquare(bitboard, square) -> bool:
    return ((bitboard >> square) & 1) != 0
