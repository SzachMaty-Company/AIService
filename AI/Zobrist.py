import os
import random
from const import *

random.seed(2137420)
import numpy as np

from Piece import Piece


class Zobrist:
    randomNumbersFileName = "./RandomNumbers.txt"

    # piece type, colour, square index
    piecesArray = np.zeros((8, 2, 64), dtype=np.uint64)
    castlingRights = [0] * 16
    # ep file (0 = no ep).
    enPassantFile = np.zeros([9], dtype=np.uint64)  # no need for rank info as side to move is included in key
    sideToMove = np.uint64(0)

    prng = random.random()

    def __init__(self):
        # randomNumbers = self.ReadRandomNumbers()

        for squareIndex in range(64):
            for pieceIndex in range(8):
                self.piecesArray[pieceIndex, WhiteIndex, squareIndex] = self.RandomUnsigned64BitNumber()
                self.piecesArray[pieceIndex, BlackIndex, squareIndex] = self.RandomUnsigned64BitNumber()

        for i in range(16):
            self.castlingRights[i] = self.RandomUnsigned64BitNumber()

        for i in range(len(self.enPassantFile)):
            self.enPassantFile[i] = self.RandomUnsigned64BitNumber()

        self.sideToMove = self.RandomUnsigned64BitNumber()

    def WriteRandomNumbers(self) -> None:
        randomNumberString = ""
        numRandomNumbers = 64 * 8 * 2 + len(self.castlingRights) + 9 + 1

        for i in range(numRandomNumbers):
            randomNumberString += str(self.RandomUnsigned64BitNumber())
            if i != numRandomNumbers - 1:
                randomNumberString += ","

        with open(self.randomNumbersFileName, "w") as file:
            file.write(randomNumberString)

    def ReadRandomNumbers(self) -> list[int]:
        if not os.path.exists(self.randomNumbersFileName):
            print("Create")
            self.WriteRandomNumbers()

        randomNumbers = []
        with open(self.randomNumbersFileName, "r") as file:
            randomNumberString = file.read()
            numberStrings = randomNumberString.split(",")

        for numberString in numberStrings:
            number = int(numberString, 16)
            randomNumbers.append(number)

        return randomNumbers

    # Calculate zobrist key from current board position. This should only be used after setting board from fen during search the key should be updated incrementally.
    def CalculateZobristKey(self, board) -> int:
        zobristKey = 0b0

        for squareIndex in range(64):
            if board.Square[squareIndex] != 0:
                pieceType = Piece.PieceType(board.Square[squareIndex])
                pieceColour = Piece.Colour(board.Square[squareIndex])

                zobristKey ^= int(
                    self.piecesArray[pieceType, WhiteIndex if pieceColour == Piece.White else BlackIndex, squareIndex]
                )

        epIndex = (board.currentGameState >> 4) & 15
        if epIndex != -1:
            zobristKey ^= int(self.enPassantFile[epIndex])

        if board.ColourToMove == Piece.Black:
            zobristKey ^= int(self.sideToMove)

        zobristKey ^= self.castlingRights[board.currentGameState & 0b1111]

        return zobristKey

    def RandomUnsigned64BitNumber(self) -> np.uint64:
        random_bytes = random.getrandbits(64).to_bytes(8, byteorder="big")
        return int.from_bytes(random_bytes, byteorder="big", signed=False)
