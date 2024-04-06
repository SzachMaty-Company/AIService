from Coord import Coord

a1 = 0
b1 = 1
c1 = 2
d1 = 3
e1 = 4
f1 = 5
g1 = 6
h1 = 7

a8 = 56
b8 = 57
c8 = 58
d8 = 59
e8 = 60
f8 = 61
g8 = 62
h8 = 63
fileNames = "abcdefgh"
rankNames = "12345678"


# Rank (0 to 7) of square
def RankIndex(squareIndex) -> int:
    return squareIndex >> 3


# File (0 to 7) of square
def FileIndex(squareIndex) -> int:
    return squareIndex & 0b000111


def IndexFromCoord(fileIndex, rankIndex) -> int:
    return rankIndex * 8 + fileIndex


# def IndexFromCoord(coord: Coord) -> int:
#     return IndexFromCoord(coord.fileIndex, coord.rankIndex)


def CoordFromIndex(squareIndex) -> Coord:
    return Coord(FileIndex(squareIndex), RankIndex(squareIndex))


def LightSquare(fileIndex, rankIndex) -> bool:
    return (fileIndex + rankIndex) % 2 != 0


def SquareNameFromCoordinate(fileIndex, rankIndex) -> str:
    return f"{fileNames[fileIndex]}{rankIndex + 1}"


def SquareNameFromIndex(squareIndex) -> str:
    return SquareNameFromSingleCoordinate(CoordFromIndex(squareIndex))


def SquareNameFromSingleCoordinate(coord: Coord) -> str:
    return SquareNameFromCoordinate(coord.fileIndex, coord.rankIndex)
