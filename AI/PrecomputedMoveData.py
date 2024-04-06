from BoardRepresentation import *
from const import *

# First 4 are orthogonal, last 4 are diagonals (N, S, W, E, NW, SE, NE, SW)
directionOffsets = (8, -8, -1, 1, 7, -7, 9, -9)

# Pawn attack directions for white and black (NW, NE SW SE)
pawnAttackDirections = ((4, 6), (7, 5))

directionLookup: list[int]

# Aka manhattan distance (answers how many moves for a rook to get from square a to square b)
orthogonalDistance = dict()
# Aka chebyshev distance (answers how many moves for a king to get from square a to square b)
kingDistance = dict()
centreManhattanDistance: list[int]


def NumRookMovesToReachSquare(startSquare, targetSquare) -> int:
    return orthogonalDistance[(startSquare, targetSquare)]


def NumKingMovesToReachSquare(startSquare, targetSquare) -> int:
    return kingDistance[(startSquare, targetSquare)]


# Initialize lookup data
pawnAttacksWhite = [0] * 64
pawnAttacksBlack = [0] * 64
knightMoves = [0] * 64
kingMoves = [0] * 64
numSquaresToEdge = [[]] * 64

rookMoves = [0] * 64
bishopMoves = [0] * 64
queenMoves = [0] * 64

# Calculate knight jumps and available squares for each square on the board.
# See comments by variable definitions for more info
allKnightJumps = (15, 17, -17, -15, 10, -6, 6, -10)
knightAttackBitboards = [0] * 64
kingAttackBitboards = [0] * 64
pawnAttackBitboards = [0] * 64

for squareIndex in range(64):
    y = int(squareIndex / 8)
    x = squareIndex - y * 8

    north = 7 - y
    south = y
    west = x
    east = 7 - x
    numSquaresToEdge[squareIndex] = [[]] * 8
    numSquaresToEdge[squareIndex][0] = north
    numSquaresToEdge[squareIndex][1] = south
    numSquaresToEdge[squareIndex][2] = west
    numSquaresToEdge[squareIndex][3] = east
    numSquaresToEdge[squareIndex][4] = min(north, west)
    numSquaresToEdge[squareIndex][5] = min(south, east)
    numSquaresToEdge[squareIndex][6] = min(north, east)
    numSquaresToEdge[squareIndex][7] = min(south, west)

    # Calculate all squares knight can jump to from current square

    legalKnightJumps = []
    knightBitboard = 0
    for knightJumpDelta in allKnightJumps:
        knightJumpSquare = squareIndex + knightJumpDelta
        if 0 <= knightJumpSquare < 64:
            knightSquareY = knightJumpSquare // 8
            knightSquareX = knightJumpSquare - knightSquareY * 8
            maxCoordMoveDst = max(abs(x - knightSquareX), abs(y - knightSquareY))
            if maxCoordMoveDst == 2:
                legalKnightJumps.append(knightJumpSquare)
                knightBitboard |= 1 << knightJumpSquare

        knightMoves[squareIndex] = legalKnightJumps
    knightAttackBitboards[squareIndex] = knightBitboard

    # Calculate all squares king can move to from current square (not including castling)
    legalKingMoves = []
    for kingMoveDelta in directionOffsets:
        kingMoveSquare = squareIndex + kingMoveDelta
        if 0 <= kingMoveSquare < 64:
            kingSquareY = kingMoveSquare // 8
            kingSquareX = kingMoveSquare - kingSquareY * 8
            maxCoordMoveDst = max(abs(x - kingSquareX), abs(y - kingSquareY))
            if maxCoordMoveDst == 1:
                legalKingMoves.append(kingMoveSquare)
                kingAttackBitboards[squareIndex] |= 1 << kingMoveSquare
    kingMoves[squareIndex] = legalKingMoves

    # Calculate legal pawn captures for white and black
    pawnCapturesWhite = []
    pawnCapturesBlack = []
    pawnAttackBitboards[squareIndex] = [0, 0]
    if x > 0:
        if y < 7:
            pawnCapturesWhite.append(squareIndex + 7)
            pawnAttackBitboards[squareIndex][WhiteIndex] |= 1 << (squareIndex + 7)
        if y > 0:
            pawnCapturesBlack.append(squareIndex - 9)
            pawnAttackBitboards[squareIndex][BlackIndex] |= 1 << (squareIndex - 9)
    if x < 7:
        if y < 7:
            pawnCapturesWhite.append(squareIndex + 9)
            pawnAttackBitboards[squareIndex][WhiteIndex] |= 1 << (squareIndex + 9)
        if y > 0:
            pawnCapturesBlack.append(squareIndex - 7)
            pawnAttackBitboards[squareIndex][BlackIndex] |= 1 << (squareIndex - 7)
    pawnAttacksWhite[squareIndex] = pawnCapturesWhite
    pawnAttacksBlack[squareIndex] = pawnCapturesBlack

    # Rook moves
    for directionIndex in range(4):
        currentDirOffset = directionOffsets[directionIndex]
        for n in range(numSquaresToEdge[squareIndex][directionIndex]):
            targetSquare = squareIndex + currentDirOffset * (n + 1)
            rookMoves[squareIndex] |= 1 << targetSquare

    # Bishop moves
    for directionIndex in range(4, 8):
        currentDirOffset = directionOffsets[directionIndex]
        for n in range(numSquaresToEdge[squareIndex][directionIndex]):
            targetSquare = squareIndex + currentDirOffset * (n + 1)
            bishopMoves[squareIndex] |= 1 << targetSquare

    queenMoves[squareIndex] = rookMoves[squareIndex] | bishopMoves[squareIndex]

directionLookup = [0] * 127
for i in range(127):
    offset = i - 63
    absOffset = abs(offset)
    absDir = 1
    if absOffset % 9 == 0:
        absDir = 9
    elif absOffset % 8 == 0:
        absDir = 8
    elif absOffset % 7 == 0:
        absDir = 7

    directionLookup[i] = absDir * (1 if offset >= 0 else -1)

# Distance lookup
centreManhattanDistance = [0] * 64
for squareA in range(64):
    coordA = CoordFromIndex(squareA)
    fileDstFromCentre = max(3 - coordA.fileIndex, coordA.fileIndex - 4)
    rankDstFromCentre = max(3 - coordA.rankIndex, coordA.rankIndex - 4)
    centreManhattanDistance[squareA] = fileDstFromCentre + rankDstFromCentre

    for squareB in range(64):
        coordB = CoordFromIndex(squareB)
        rankDistance = abs(coordA.rankIndex - coordB.rankIndex)
        fileDistance = abs(coordA.fileIndex - coordB.fileIndex)
        orthogonalDistance[(squareA, squareB)] = fileDistance + rankDistance
        kingDistance[(squareA, squareB)] = max(fileDistance, rankDistance)
