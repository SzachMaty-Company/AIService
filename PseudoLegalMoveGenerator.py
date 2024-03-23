import numpy as np

from Move import Move
from Piece import Piece
from PieceList import PieceList
from Board import Board
from PrecomputedMoveData import *
from BoardRepresentation import *
from const import *


class PseudoLegalMoveGenerator:
    moves: list[Move]
    isWhiteToMove = True
    friendlyColour: int
    opponentColour: int
    friendlyKingSquare: int
    friendlyColourIndex: int
    opponentColourIndex: int

    genQuiets = True
    genUnderpromotions = True
    board: Board

    def __init__(self, board: Board):
        self.moves = []
        self.isWhiteToMove = board.ColourToMove == Piece.White
        self.friendlyColour = board.ColourToMove
        self.opponentColour = board.OpponentColour
        self.friendlyKingSquare = board.KingSquare[board.ColourToMoveIndex]
        self.friendlyColourIndex = WhiteIndex if board.WhiteToMove else BlackIndex
        self.opponentColourIndex = 1 - self.friendlyColourIndex
        self.board = board

    def GenerateMoves(self, includeQuietMoves=True, includeUnderPromotions=True) -> list[Move]:
        self.genQuiets = includeQuietMoves
        self.genUnderpromotions = includeUnderPromotions
        # s = len(self.moves)
        self.GenerateKingMoves()
        # print("king moves", len(self.moves) - s)
        # s = len(self.moves)
        self.GenerateSlidingMoves()
        # print("sliding moves", len(self.moves) - s)
        # s = len(self.moves)
        self.GenerateKnightMoves()
        # print("knight moves", len(self.moves) - s)
        # s = len(self.moves)
        self.GeneratePawnMoves()
        # print("pawn moves", len(self.moves) - s)

        return self.moves

    def SquareAttacked(self, attackSquare, attackerColour) -> bool:
        attackerColourIndex = WhiteIndex if attackerColour == Piece.White else BlackIndex
        self.friendlyColourIndex = 1 - attackerColourIndex
        self.friendlyColour = Piece.Black if attackerColour == Piece.White else Piece.White

        startDirIndex = 0
        endDirIndex = 8

        opponentKingSquare = self.board.KingSquare[attackerColourIndex]
        if kingDistance[opponentKingSquare, attackSquare] == 1:
            return True

        if self.board.queens[attackerColourIndex].Count == 0:
            startDirIndex = 0 if len(self.board.rooks[attackerColourIndex]) else 4
            endDirIndex = 8 if len(self.board.rooks[attackerColourIndex]) else 4

        for dir in range(startDirIndex, endDirIndex):
            isDiagonal = dir > 3

            n = numSquaresToEdge[attackSquare][dir]
            directionOffset = directionOffsets[dir]

            for i in range(n):
                squareIndex = attackSquare + directionOffset * (i + 1)
                piece = self.board.Square[squareIndex]

                # This square contains a piece
                if piece != Piece.Empty:
                    if Piece.IsColour(piece, self.friendlyColour):
                        break
                    # This square contains an enemy piece
                    else:
                        pieceType = Piece.PieceType(piece)

                        # Check if piece is in bitmask of pieces able to move in current direction
                        if (
                            isDiagonal
                            and Piece.IsBishopOrQueen(pieceType)
                            or not isDiagonal
                            and Piece.IsRookOrQueen(pieceType)
                        ):
                            return True
                        else:
                            # This enemy piece is not able to move in the current direction, and so is blocking any checks/pins
                            break

        # Knight attacks
        knightAttackSquares = knightMoves[attackSquare]
        for i in range(len(knightAttackSquares)):
            if self.board.Square[knightAttackSquares[i]] == (Piece.Knight or attackerColour):
                return True

        # check if enemy pawn is controlling this square
        for i in range(2):
            # Check if square exists diagonal to friendly king from which enemy pawn could be attacking it
            if numSquaresToEdge[attackSquare][pawnAttackDirections[self.friendlyColourIndex][i]] > 0:
                # move in direction friendly pawns attack to get square from which enemy pawn would attack
                s = attackSquare + directionOffsets[pawnAttackDirections[self.friendlyColourIndex][i]]

                piece = self.board.Square[s]
                if piece == (Piece.Pawn or attackerColour):  # is enemy pawn
                    return True

        return False

    def GenerateSlidingMoves(self) -> None:
        rooks: PieceList = self.board.rooks[self.friendlyColourIndex]
        for rook in rooks:
            if rook is not None:
                self.GenerateSlidingPieceMoves(rook, 0, 4)

        bishops: PieceList = self.board.bishops[self.friendlyColourIndex]
        for bishop in bishops:
            if bishop is not None:
                self.GenerateSlidingPieceMoves(bishop, 4, 8)

        queens: PieceList = self.board.queens[self.friendlyColourIndex]
        for queen in queens:
            if queen is not None:
                self.GenerateSlidingPieceMoves(queen, 0, 8)

    def GenerateSlidingPieceMoves(self, startSquare, startDirIndex, endDirIndex) -> None:
        for directionIndex in range(startDirIndex, endDirIndex):
            currentDirOffset = directionOffsets[directionIndex]

            for n in range(numSquaresToEdge[startSquare][directionIndex]):
                targetSquare = startSquare + currentDirOffset * (n + 1)
                targetSquarePiece = self.board.Square[targetSquare]

                # Blocked by friendly piece, so stop looking in this direction
                if Piece.IsColour(targetSquarePiece, self.friendlyColour):
                    break

                isCapture = targetSquarePiece != Piece.Empty

                if self.genQuiets or isCapture:
                    self.moves.append(Move(startSquare, targetSquare))

                # If square not empty, can't move any further in this direction
                # Also, if this move blocked a check, further moves won't block the check
                if isCapture:
                    break

    def GenerateKnightMoves(self) -> None:
        myKnights: PieceList = self.board.knights[self.friendlyColourIndex]

        for startSquare in myKnights:
            if startSquare is None:
                continue
            for targetSquare in knightMoves[startSquare]:
                targetSquarePiece = self.board.Square[targetSquare]
                isCapture = Piece.IsColour(targetSquarePiece, self.opponentColour)
                if self.genQuiets or isCapture:
                    # Skip if square contains friendly piece, or if in check and knight is not interposing/capturing checking piece
                    if Piece.IsColour(targetSquarePiece, self.friendlyColour):
                        continue
                    self.moves.append(Move(startSquare, targetSquare))

    def MakePromotionMoves(self, fromSquare, toSquare) -> None:
        self.moves.append(Move(fromSquare, toSquare, Move.Flag.PromoteToQueen))
        if self.genUnderpromotions:
            self.moves.append(Move(fromSquare, toSquare, Move.Flag.PromoteToKnight))
            self.moves.append(Move(fromSquare, toSquare, Move.Flag.PromoteToRook))
            self.moves.append(Move(fromSquare, toSquare, Move.Flag.PromoteToBishop))

    def GenerateKingMoves(self) -> None:
        for targetSquare in kingMoves[self.friendlyKingSquare]:
            pieceOnTargetSquare = self.board.Square[targetSquare]

            # Skip squares occupied by friendly pieces
            if Piece.IsColour(pieceOnTargetSquare, self.friendlyColour):
                continue

            isCapture = Piece.IsColour(pieceOnTargetSquare, self.opponentColour)
            if not isCapture:
                # King can't move to square marked as under enemy control, unless he is capturing that piece
                # Also skip if not generating quiet moves
                if not self.genQuiets:
                    continue

            # Safe for king to move to this square

            self.moves.append(Move(self.friendlyKingSquare, targetSquare))

            # Castling:
            if not isCapture and not self.SquareAttacked(self.friendlyKingSquare, self.opponentColour):
                # Castle kingside
                if (targetSquare == f1 or targetSquare == f8) and self.HasKingsideCastleRight():
                    if not self.SquareAttacked(targetSquare, self.opponentColour):
                        castleKingsideSquare = targetSquare + 1
                        if self.board.Square[castleKingsideSquare] == Piece.Empty:
                            self.moves.append(Move(self.friendlyKingSquare, castleKingsideSquare, Move.Flag.Castling))

                # Castle queenside
                elif (targetSquare == d1 or targetSquare == d8) and self.HasQueensideCastleRight():
                    if not self.SquareAttacked(targetSquare, self.opponentColour):
                        castleQueensideSquare = targetSquare - 1
                        if (
                            self.board.Square[castleQueensideSquare] == Piece.Empty
                            and self.board.Square[castleQueensideSquare - 1] == Piece.Empty
                        ):
                            self.moves.append(Move(self.friendlyKingSquare, castleQueensideSquare, Move.Flag.Castling))

    def GeneratePawnMoves(self) -> None:
        myPawns: PieceList = self.board.pawns[self.friendlyColourIndex]
        pawnOffset = 8 if self.friendlyColour == Piece.White else -8
        startRank = 1 if self.board.WhiteToMove else 6
        finalRankBeforePromotion = 6 if self.board.WhiteToMove else 1

        enPassantFile = ((self.board.currentGameState >> 4) & 15) - 1
        enPassantSquare = -1
        if enPassantFile != -1:
            enPassantSquare = 8 * (5 if self.board.WhiteToMove else 2) + enPassantFile

        for startSquare in myPawns:
            if startSquare is None:
                continue
            rank = RankIndex(startSquare)
            oneStepFromPromotion = rank == finalRankBeforePromotion

            if self.genQuiets:
                squareOneForward = startSquare + pawnOffset

                # Square ahead of pawn is empty: forward moves
                if self.board.Square[squareOneForward] == Piece.Empty:
                    # Pawn not pinned, or is moving along line of pin

                    if oneStepFromPromotion:
                        self.MakePromotionMoves(startSquare, squareOneForward)
                    else:
                        self.moves.append(Move(startSquare, squareOneForward))

                    # Is on starting square (so can move two forward if not blocked)
                    if rank == startRank:
                        squareTwoForward = squareOneForward + pawnOffset
                        if self.board.Square[squareTwoForward] == Piece.Empty:
                            # Not in check, or pawn is interposing checking piece

                            self.moves.append(Move(startSquare, squareTwoForward, Move.Flag.PawnTwoForward))

            # Pawn captures
            for j in range(2):
                # Check if square exists diagonal to pawn
                if numSquaresToEdge[startSquare][pawnAttackDirections[self.friendlyColourIndex][j]] > 0:
                    # move in direction friendly pawns attack to get square from which enemy pawn would attack
                    pawnCaptureDir = directionOffsets[pawnAttackDirections[self.friendlyColourIndex][j]]
                    targetSquare = startSquare + pawnCaptureDir
                    targetPiece = self.board.Square[targetSquare]

                    # Regular capture
                    if Piece.IsColour(targetPiece, self.opponentColour):
                        if oneStepFromPromotion:
                            self.MakePromotionMoves(startSquare, targetSquare)
                        else:
                            self.moves.append(Move(startSquare, targetSquare))

                    # Capture en-passant
                    if targetSquare == enPassantSquare:
                        epCapturedPawnSquare = targetSquare + (-8 if self.board.WhiteToMove else 8)

                        self.moves.append(Move(startSquare, targetSquare, Move.Flag.EnPassantCapture))

    def InCheck(self) -> bool:
        return False
        # return self.SquareAttacked (friendlyKingSquare, board.ColourToMoveIndex)

    def Illegal(self) -> bool:
        return self.SquareAttacked(self.board.KingSquare[1 - self.board.ColourToMoveIndex], self.board.ColourToMove)

    def HasKingsideCastleRight(self) -> bool:
        mask = 1 if self.board.WhiteToMove else 4
        return (self.board.currentGameState & mask) != 0

    def HasQueensideCastleRight(self) -> bool:
        mask = 2 if self.board.WhiteToMove else 8
        return (self.board.currentGameState & mask) != 0
