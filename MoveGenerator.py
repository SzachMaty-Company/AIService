from Move import Move
from BoardRepresentation import *
from Board import Board
from Piece import Piece
from const import *
from PieceList import PieceList
from PrecomputedMoveData import *
from BitBoardUtility import ContainsSquare


class MoveGenerator:
    class PromotionMode:
        All = 0
        QueenOnly = 1
        QueenAndKnight = 2

    promotionsToGenerate = PromotionMode.All

    # ---- Instance variables ----
    moves: list[Move]
    isWhiteToMove: bool
    friendlyColour: int
    opponentColour: int
    friendlyKingSquare: int
    friendlyColourIndex: int
    opponentColourIndex: int

    inCheck: bool
    inDoubleCheck: bool
    pinsExistInPosition: bool
    checkRayBitmask: int
    pinRayBitmask: int
    opponentKnightAttacks: int
    opponentAttackMapNoPawns: int
    opponentAttackMap: int
    opponentPawnAttackMap: int
    opponentSlidingAttackMap: int

    genQuiets: bool
    board: Board

    def Init(self, board: Board):
        self.moves = []
        self.inCheck = False
        self.inDoubleCheck = False
        self.pinsExistInPosition = False
        self.checkRayBitmask = 0b0
        self.pinRayBitmask = 0b0
        self.isWhiteToMove = board.ColourToMove == Piece.White
        self.friendlyColour = board.ColourToMove
        self.opponentColour = board.OpponentColour
        self.friendlyKingSquare = board.KingSquare[board.ColourToMoveIndex]
        self.friendlyColourIndex = WhiteIndex if board.WhiteToMove else BlackIndex
        self.opponentColourIndex = 1 - self.friendlyColourIndex
        self.board = board

    # Generates list of legal moves in current position.
    # Quiet moves (non captures) can optionally be excluded. This is used in quiescence search.
    def GenerateMoves(self, board:Board, includeQuietMoves=True) -> list[Move]:
        self.Init(board)
        self.genQuiets = includeQuietMoves
        self.CalculateAttackData()
        # print("start moves", len(self.moves))
        # s = len(self.moves)
        self.GenerateKingMoves()
        if self.inDoubleCheck:
            return self.moves
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
                if not self.genQuiets or self.SquareIsInCheckRay(targetSquare):
                    continue

            # Safe for king to move to this square
            if not self.SquareIsAttacked(targetSquare):
                self.moves.append(Move(self.friendlyKingSquare, targetSquare))

                # Castling:
                if not isCapture and not isCapture:
                    # Castle kingside
                    if (targetSquare == f1 or targetSquare == f8) and self.HasKingsideCastleRight():
                        castleKingsideSquare = targetSquare + 1
                        if self.board.Square[castleKingsideSquare] == Piece.Empty:
                            if not self.SquareIsAttacked(castleKingsideSquare):
                                self.moves.append(
                                    Move(self.friendlyKingSquare, castleKingsideSquare, Move.Flag.Castling)
                                )

                    # Castle queenside
                    elif (targetSquare == d1 or targetSquare == d8) and self.HasQueensideCastleRight():
                        castleQueensideSquare = targetSquare - 1
                        if (
                            self.board.Square[castleQueensideSquare] == Piece.Empty
                            and self.board.Square[castleQueensideSquare - 1] == Piece.Empty
                        ):
                            if not self.SquareIsAttacked(castleQueensideSquare):
                                self.moves.append(
                                    Move(self.friendlyKingSquare, castleQueensideSquare, Move.Flag.Castling)
                                )

    def GenerateSlidingMoves(self) -> None:
        rooks: PieceList = self.board.rooks[self.friendlyColourIndex]
        for rook in rooks:
            self.GenerateSlidingPieceMoves(rook, 0, 4)

        bishops: PieceList = self.board.bishops[self.friendlyColourIndex]
        for bishop in bishops:
            self.GenerateSlidingPieceMoves(bishop, 4, 8)

        queens: PieceList = self.board.queens[self.friendlyColourIndex]
        for queen in queens:
            self.GenerateSlidingPieceMoves(queen, 0, 8)

    def GenerateSlidingPieceMoves(self, startSquare, startDirIndex, endDirIndex) -> None:
        isPinned = self.IsPinned(startSquare)

        # If this piece is pinned, and the king is in check, this piece cannot move
        if self.inCheck and isPinned:
            return

        for directionIndex in range(startDirIndex, endDirIndex):
            currentDirOffset = directionOffsets[directionIndex]

            # If pinned, this piece can only move along the ray towards/away from the friendly king, so skip other directions
            if isPinned and not self.IsMovingAlongRay(currentDirOffset, self.friendlyKingSquare, startSquare):
                continue

            for n in range(numSquaresToEdge[startSquare][directionIndex]):
                targetSquare = startSquare + currentDirOffset * (n + 1)
                targetSquarePiece = self.board.Square[targetSquare]

                # Blocked by friendly piece, so stop looking in this direction
                if Piece.IsColour(targetSquarePiece, self.friendlyColour):
                    break

                isCapture = targetSquarePiece != Piece.Empty

                movePreventsCheck = self.SquareIsInCheckRay(targetSquare)

                if movePreventsCheck or not self.inCheck:
                    if self.genQuiets or isCapture:
                        self.moves.append(Move(startSquare, targetSquare))

                # If square not empty, can't move any further in this direction
                # Also, if this move blocked a check, further moves won't block the check
                if isCapture or movePreventsCheck:
                    break

    def GenerateKnightMoves(self) -> None:
        myKnights: PieceList = self.board.knights[self.friendlyColourIndex]

        for startSquare in myKnights:
            if self.IsPinned(startSquare):
                continue
            for targetSquare in knightMoves[startSquare]:
                targetSquarePiece = self.board.Square[targetSquare]
                isCapture = Piece.IsColour(targetSquarePiece, self.opponentColour)
                if self.genQuiets or isCapture:
                    # Skip if square contains friendly piece, or if in check and knight is not interposing/capturing checking piece
                    if (
                        Piece.IsColour(targetSquarePiece, self.friendlyColour)
                        or self.inCheck
                        and not self.SquareIsInCheckRay(targetSquare)
                    ):
                        continue
                    self.moves.append(Move(startSquare, targetSquare))

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
            rank = RankIndex(startSquare)
            oneStepFromPromotion = rank == finalRankBeforePromotion

            if self.genQuiets:
                squareOneForward = startSquare + pawnOffset
                # Square ahead of pawn is empty: forward moves
                if self.board.Square[squareOneForward] == Piece.Empty:
                    # Pawn not pinned, or is moving along line of pin
                    if not self.IsPinned(startSquare) or self.IsMovingAlongRay(
                        pawnOffset, startSquare, self.friendlyKingSquare
                    ):
                        # Not in check, or pawn is interposing checking piece
                        if not self.inCheck or self.SquareIsInCheckRay(squareOneForward):
                            if oneStepFromPromotion:
                                self.MakePromotionMoves(startSquare, squareOneForward)
                            else:
                                self.moves.append(Move(startSquare, squareOneForward))

                    # Is on starting square (so can move two forward if not blocked)
                    if rank == startRank:
                        squareTwoForward = squareOneForward + pawnOffset
                        if self.board.Square[squareTwoForward] == Piece.Empty:
                            # Not in check, or pawn is interposing checking piece
                            if not self.inCheck or self.SquareIsInCheckRay(squareTwoForward):
                                self.moves.append(Move(startSquare, squareTwoForward, Move.Flag.PawnTwoForward))

            # Pawn captures
            for j in range(2):
                # Check if square exists diagonal to pawn
                if numSquaresToEdge[startSquare][pawnAttackDirections[self.friendlyColourIndex][j]] > 0:
                    # move in direction friendly pawns attack to get square from which enemy pawn would attack
                    pawnCaptureDir = directionOffsets[pawnAttackDirections[self.friendlyColourIndex][j]]
                    targetSquare = startSquare + pawnCaptureDir
                    targetPiece = self.board.Square[targetSquare]

                    # If piece is pinned, and the square it wants to move to is not on same line as the pin, then skip this direction
                    if self.IsPinned(startSquare) and not self.IsMovingAlongRay(
                        pawnCaptureDir, self.friendlyKingSquare, startSquare
                    ):
                        continue

                    # Regular capture
                    if Piece.IsColour(targetPiece, self.opponentColour):
                        # If in check, and piece is not capturing/interposing the checking piece, then skip to next square
                        if self.inCheck and not self.SquareIsInCheckRay(targetSquare):
                            continue
                        if oneStepFromPromotion:
                            self.MakePromotionMoves(startSquare, targetSquare)
                        else:
                            self.moves.append(Move(startSquare, targetSquare))

                    # Capture en-passant
                    if targetSquare == enPassantSquare:
                        epCapturedPawnSquare = targetSquare + (-8 if self.board.WhiteToMove else 8)
                        if not self.InCheckAfterEnPassant(startSquare, targetSquare, epCapturedPawnSquare):
                            self.moves.append(Move(startSquare, targetSquare, Move.Flag.EnPassantCapture))

    def MakePromotionMoves(self, fromSquare, toSquare) -> None:
        self.moves.append(Move(fromSquare, toSquare, Move.Flag.PromoteToQueen))
        if self.promotionsToGenerate == self.PromotionMode.All:
            self.moves.append(Move(fromSquare, toSquare, Move.Flag.PromoteToKnight))
            self.moves.append(Move(fromSquare, toSquare, Move.Flag.PromoteToRook))
            self.moves.append(Move(fromSquare, toSquare, Move.Flag.PromoteToBishop))
        elif self.promotionsToGenerate == self.PromotionMode.QueenAndKnight:
            self.moves.append(Move(fromSquare, toSquare, Move.Flag.PromoteToKnight))

    def IsMovingAlongRay(self, rayDir, startSquare, targetSquare) -> bool:
        moveDir = directionLookup[targetSquare - startSquare + 63]
        return rayDir == moveDir or -rayDir == moveDir

    def IsPinned(self, square) -> bool:
        return self.pinsExistInPosition and ((self.pinRayBitmask >> square) & 1) != 0

    def SquareIsInCheckRay(self, square) -> bool:
        return self.inCheck and ((self.checkRayBitmask >> square) & 1) != 0

    def HasKingsideCastleRight(self) -> bool:
        mask = 1 if self.board.WhiteToMove else 4
        return (self.board.currentGameState & mask) != 0

    def HasQueensideCastleRight(self) -> bool:
        mask = 2 if self.board.WhiteToMove else 8
        return (self.board.currentGameState & mask) != 0

    def GenSlidingAttackMap(self) -> None:
        self.opponentSlidingAttackMap = 0

        enemyRooks: PieceList = self.board.rooks[self.opponentColourIndex]
        for rook in enemyRooks:
            self.UpdateSlidingAttackPiece(rook, 0, 4)

        enemyQueens: PieceList = self.board.queens[self.opponentColourIndex]
        for queen in enemyQueens:
            self.UpdateSlidingAttackPiece(queen, 0, 8)

        enemyBishops: PieceList = self.board.bishops[self.opponentColourIndex]
        for bishop in enemyBishops:
            self.UpdateSlidingAttackPiece(bishop, 4, 8)

    def UpdateSlidingAttackPiece(self, startSquare, startDirIndex, endDirIndex) -> None:
        for directionIndex in range(startDirIndex, endDirIndex):
            currentDirOffset = directionOffsets[directionIndex]
            for n in range(numSquaresToEdge[startSquare][directionIndex]):
                targetSquare = startSquare + currentDirOffset * (n + 1)
                targetSquarePiece = self.board.Square[targetSquare]
                self.opponentSlidingAttackMap |= 1 << targetSquare
                if targetSquare != self.friendlyKingSquare:
                    if targetSquarePiece != Piece.Empty:
                        break

    def CalculateAttackData(self) -> None:
        self.GenSlidingAttackMap()
        startDirIndex = 0
        endDirIndex = 8

        if len(self.board.queens[self.opponentColourIndex]) == 0:
            startDirIndex = 0 if len(self.board.rooks[self.opponentColourIndex]) > 0 else 4
            endDirIndex = 8 if len(self.board.bishops[self.opponentColourIndex]) > 0 else 4

        for dir in range(startDirIndex, endDirIndex):
            isDiagonal = dir > 3
            n = numSquaresToEdge[self.friendlyKingSquare][dir]
            directionOffset = directionOffsets[dir]
            isFriendlyPieceAlongRay = False
            rayMask = 0b0
            for i in range(n):
                squareIndex = self.friendlyKingSquare + directionOffset * (i + 1)
                rayMask |= 1 << squareIndex
                piece = self.board.Square[squareIndex]

                if piece != Piece.Empty:
                    if Piece.IsColour(piece, self.friendlyColour):
                        if not isFriendlyPieceAlongRay:
                            isFriendlyPieceAlongRay = True
                        else:
                            break
                    else:
                        pieceType = Piece.PieceType(piece)
                        # Check if piece is in bitmask of pieces able to move in current direction
                        if (isDiagonal and Piece.IsBishopOrQueen(pieceType)) or (
                            not isDiagonal and Piece.IsRookOrQueen(pieceType)
                        ):
                            if isFriendlyPieceAlongRay:
                                self.pinsExistInPosition = True
                                self.pinRayBitmask |= rayMask
                            else:
                                self.checkRayBitmask |= rayMask
                                self.inDoubleCheck = self.inCheck
                                self.inCheck = True
                            break
                        else:
                            break
            if self.inDoubleCheck:
                break

        opponentKnights: PieceList = self.board.knights[self.opponentColourIndex]
        self.opponentKnightAttacks = 0
        isKnightCheck = False

        for startSquare in opponentKnights:
            self.opponentKnightAttacks |= knightAttackBitboards[startSquare]
            if not isKnightCheck and ContainsSquare(self.opponentKnightAttacks, self.friendlyKingSquare):
                isKnightCheck = True
                self.inDoubleCheck = self.inCheck
                self.inCheck = True
                self.checkRayBitmask |= 0b1 << startSquare
        # Pawn attacks
        opponentPawns: PieceList = self.board.pawns[self.opponentColourIndex]
        self.opponentPawnAttackMap = 0
        isPawnCheck = False

        for pawnSquare in opponentPawns:
            pawnAttacks = pawnAttackBitboards[pawnSquare][self.opponentColourIndex]
            self.opponentPawnAttackMap |= pawnAttacks

            if not isPawnCheck and ContainsSquare(pawnAttacks, self.friendlyKingSquare):
                isPawnCheck = True
                self.inDoubleCheck = self.inCheck  # if already in check, then this is double check
                self.inCheck = True
                self.checkRayBitmask |= 0b1 << pawnSquare

        enemyKingSquare = self.board.KingSquare[self.opponentColourIndex]
        self.opponentAttackMapNoPawns = (
            self.opponentSlidingAttackMap | self.opponentKnightAttacks | kingAttackBitboards[enemyKingSquare]
        )
        self.opponentAttackMap = self.opponentAttackMapNoPawns | self.opponentPawnAttackMap

    def SquareIsAttacked(self, square) -> bool:
        return ContainsSquare(self.opponentAttackMap, square)

    def InCheckAfterEnPassant(self, startSquare, targetSquare, epCapturedPawnSquare) -> bool:
        # Update board to reflect en-passant capture
        self.board.Square[targetSquare] = self.board.Square[startSquare]
        self.board.Square[startSquare] = Piece.Empty
        self.board.Square[epCapturedPawnSquare] = Piece.Empty

        inCheckAfterEpCapture = False
        if self.SquareAttackedAfterEPCapture(epCapturedPawnSquare, startSquare):
            inCheckAfterEpCapture = True

        # Undo change to board
        self.board.Square[targetSquare] = Piece.Empty
        self.board.Square[startSquare] = Piece.Pawn | self.friendlyColour
        self.board.Square[epCapturedPawnSquare] = Piece.Pawn | self.opponentColour
        return inCheckAfterEpCapture

    def SquareAttackedAfterEPCapture(self, epCaptureSquare, capturingPawnStartSquare) -> bool:
        if ContainsSquare(self.opponentAttackMapNoPawns, self.friendlyKingSquare):
            return True

        # Loop through the horizontal direction towards ep capture to see if any enemy piece now attacks king
        dirIndex = 2 if epCaptureSquare < self.friendlyKingSquare else 3

        for i in range(numSquaresToEdge[epCaptureSquare][dirIndex]):
            squareIndex = epCaptureSquare + directionOffsets[dirIndex] * (i + 1)
            piece = self.board.Square[squareIndex]
            if piece != Piece.Empty:
                # Friendly piece is blocking view of this square from the enemy.
                if Piece.IsColour(piece, self.friendlyColour):
                    break
                # This square contains an enemy piece
                else:
                    if Piece.IsRookOrQueen(piece):
                        return True
                    else:
                        # This piece is not able to move in the current direction, and is therefore blocking any checks along this line
                        break

        # check if enemy pawn is controlling this square (can't use pawn attack bitboard, because pawn has been captured)
        for i in range(2):
            if numSquaresToEdge[epCaptureSquare][pawnAttackDirections[self.friendlyColourIndex][i]] > 0:
                squareIndex = epCaptureSquare + directionOffsets[pawnAttackDirections[self.friendlyColourIndex][i]]
                piece = self.board.Square[squareIndex]
                if piece == (Piece.Pawn | self.opponentColour):
                    return True
        return False
