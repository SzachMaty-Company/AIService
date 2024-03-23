class PieceList:
    # Indices of squares occupied by given piece type (only elements up to Count are valid, the rest are unused/garbage)
    occupiedSquares: list[int]
    # piece_map to go from index of a square, to the index in the occupiedSquares array where that square is stored
    piece_map: list[int]
    numPieces: int

    def __init__(self, maxPieceCount=16):
        self.occupiedSquares = []
        self.piece_map = [0] * 64
        self.numPieces = 0

    def Count(self) -> int:
        return self.numPieces

    def AddPieceAtSquare(self, square) -> None:
        self.occupiedSquares.append(square)
        self.piece_map[square] = self.numPieces
        self.numPieces += 1

    def RemovePieceAtSquare(self, square) -> None:
        pieceIndex = self.piece_map[square]  # get the index of this element in the occupiedSquares array
        if pieceIndex == self.numPieces - 1:
            self.piece_map[square] = 0
            self.occupiedSquares.pop()
        else:
            self.occupiedSquares[pieceIndex] = self.occupiedSquares[
                self.numPieces - 1
            ]  # move last element in array to the place of the removed element
            self.piece_map[
                self.occupiedSquares[pieceIndex]
            ] = pieceIndex  # update piece_map to point to the moved element's new location in the array
            self.occupiedSquares.pop()
        self.numPieces -= 1

    def MovePiece(self, startSquare, targetSquare) -> None:
        pieceIndex = self.piece_map[startSquare]  # get the index of this element in the occupiedSquares array
        self.occupiedSquares[pieceIndex] = targetSquare
        self.piece_map[targetSquare] = pieceIndex

    def __iter__(self):
        return self.occupiedSquares.__iter__()

    def __getitem__(self, index):
        return self.occupiedSquares[index]

    def __len__(self):
        return self.numPieces
