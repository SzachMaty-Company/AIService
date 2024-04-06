from PieceList import PieceList
from Move import Move
from Piece import Piece
from BoardRepresentation import *
from Zobrist import Zobrist
from FenUtility import FenUtility

from const import *


class Board:
    # Stores piece code for each square on the board.
    # Piece code is defined as piecetype | colour code
    Square: list[int]

    WhiteToMove = True
    ColourToMove: int
    OpponentColour: int
    ColourToMoveIndex: int

    # Bits 0-3 store white and black kingside/queenside castling legality
    # Bits 4-7 store file of ep square (starting at 1, so 0 = no ep square)
    # Bits 8-13 captured piece
    # Bits 14-... fifty mover counter
    gameStateHistory: list[int]
    currentGameState = 0

    plyCount: int  # // Total plies played in game
    fiftyMoveCounter: int  # // Num ply since last pawn move or capture

    ZobristKey: int
    # List of zobrist keys
    RepetitionPositionHistory: list[int]

    KingSquare: list[int]  # index of square of white and black king

    rooks: list[PieceList]
    bishops: list[PieceList]
    queens: list[PieceList]
    knights: list[PieceList]
    pawns: list[PieceList]

    allPieceLists: list[PieceList]

    whiteCastleKingsideMask = 0b1111111111111110
    whiteCastleQueensideMask = 0b1111111111111101
    blackCastleKingsideMask = 0b1111111111111011
    blackCastleQueensideMask = 0b1111111111110111

    whiteCastleMask = whiteCastleKingsideMask & whiteCastleQueensideMask
    blackCastleMask = blackCastleKingsideMask & blackCastleQueensideMask

    def __init__(self):
        self.Square = [0] * 64
        self.KingSquare = [0, 0]

        self.gameStateHistory = []
        self.ZobristKey = 0b0
        self.RepetitionPositionHistory = []
        self.plyCount = 0
        self.fiftyMoveCounter = 0

        self.knights = [PieceList(10), PieceList(10)]
        self.pawns = [PieceList(8), PieceList(8)]
        self.rooks = [PieceList(10), PieceList(10)]
        self.bishops = [PieceList(10), PieceList(10)]
        self.queens = [PieceList(9), PieceList(9)]
        emptyList = PieceList(0)
        self.allPieceLists = [
            emptyList,
            emptyList,
            self.pawns[WhiteIndex],
            self.knights[WhiteIndex],
            emptyList,
            self.bishops[WhiteIndex],
            self.rooks[WhiteIndex],
            self.queens[WhiteIndex],
            emptyList,
            emptyList,
            self.pawns[BlackIndex],
            self.knights[BlackIndex],
            emptyList,
            self.bishops[BlackIndex],
            self.rooks[BlackIndex],
            self.queens[BlackIndex],
        ]

    def GetPieceList(self, pieceType, colourIndex) -> PieceList:
        return self.allPieceLists[colourIndex * 8 + pieceType]

    # Make a move on the board
    # The inSearch parameter controls whether this move should be recorded in the game history (for detecting three-fold repetition)
    def MakeMove(self, move: Move, inSearch=False) -> bool:
        oldEnPassantFile = (self.currentGameState >> 4) & 15
        originalCastleState = self.currentGameState & 15
        newCastleState = originalCastleState
        self.currentGameState = 0b0

        opponentColourIndex = 1 - self.ColourToMoveIndex
        moveFrom = move.StartSquare()
        moveTo = move.TargetSquare()

        capturedPieceType = Piece.PieceType(self.Square[moveTo])
        was_capture = capturedPieceType != Piece.Empty
        movePiece = self.Square[moveFrom]
        movePieceType = Piece.PieceType(movePiece)

        moveFlag = move.MoveFlag()
        isPromotion = move.IsPromotion()
        isEnPassant = moveFlag == Move.Flag.EnPassantCapture

        # Handle captures
        self.currentGameState |= capturedPieceType << 8
        if capturedPieceType != 0 and not isEnPassant:
            self.ZobristKey ^= int(Zobrist.piecesArray[capturedPieceType, opponentColourIndex, moveTo])
            self.GetPieceList(capturedPieceType, opponentColourIndex).RemovePieceAtSquare(moveTo)

        # Move pieces in piece lists
        if movePieceType == Piece.King:
            self.KingSquare[self.ColourToMoveIndex] = moveTo
            newCastleState &= self.whiteCastleMask if self.WhiteToMove else self.blackCastleMask
        else:
            self.GetPieceList(movePieceType, self.ColourToMoveIndex).MovePiece(moveFrom, moveTo)

        pieceOnTargetSquare = movePiece

        # Handle promotion
        if isPromotion:
            promoteType = 0
            match moveFlag:
                case Move.Flag.PromoteToQueen:
                    promoteType = Piece.Queen
                    self.queens[self.ColourToMoveIndex].AddPieceAtSquare(moveTo)
                case Move.Flag.PromoteToRook:
                    promoteType = Piece.Rook
                    self.rooks[self.ColourToMoveIndex].AddPieceAtSquare(moveTo)
                case Move.Flag.PromoteToBishop:
                    promoteType = Piece.Bishop
                    self.bishops[self.ColourToMoveIndex].AddPieceAtSquare(moveTo)
                case Move.Flag.PromoteToKnight:
                    promoteType = Piece.Knight
                    self.knights[self.ColourToMoveIndex].AddPieceAtSquare(moveTo)

            pieceOnTargetSquare = promoteType | self.ColourToMove
            self.pawns[self.ColourToMoveIndex].RemovePieceAtSquare(moveTo)
        else:
            # Handle other special moves (en-passant, and castling)
            match moveFlag:
                case Move.Flag.EnPassantCapture:
                    epPawnSquare = moveTo + (-8 if self.ColourToMove == Piece.White else 8)
                    self.currentGameState |= self.Square[epPawnSquare] << 8  # add pawn as capture type
                    self.Square[epPawnSquare] = 0  # clear ep capture square
                    self.pawns[opponentColourIndex].RemovePieceAtSquare(epPawnSquare)
                    self.ZobristKey ^= int(Zobrist.piecesArray[Piece.Pawn, opponentColourIndex, epPawnSquare])
                    was_capture = True
                case Move.Flag.Castling:
                    kingside = moveTo == g1 or moveTo == g8
                    castlingRookFromIndex = moveTo + 1 if kingside else moveTo - 2
                    castlingRookToIndex = moveTo - 1 if kingside else moveTo + 1

                    self.Square[castlingRookFromIndex] = Piece.Empty
                    self.Square[castlingRookToIndex] = Piece.Rook | self.ColourToMove

                    self.rooks[self.ColourToMoveIndex].MovePiece(castlingRookFromIndex, castlingRookToIndex)
                    self.ZobristKey ^= int(Zobrist.piecesArray[Piece.Rook, self.ColourToMoveIndex, castlingRookFromIndex])
                    self.ZobristKey ^= int(Zobrist.piecesArray[Piece.Rook, self.ColourToMoveIndex, castlingRookToIndex])

        # Update the board representation:
        self.Square[moveTo] = pieceOnTargetSquare
        self.Square[moveFrom] = 0

        # Pawn has moved two forwards, mark file with en-passant flag
        if moveFlag == Move.Flag.PawnTwoForward:
            file = FileIndex(moveFrom) + 1
            self.currentGameState |= file << 4
            self.ZobristKey ^= int(Zobrist.enPassantFile[file])

        # Piece moving to/from rook square removes castling right for that side
        if originalCastleState != 0:
            if moveTo == h1 or moveFrom == h1:
                newCastleState &= self.whiteCastleKingsideMask
            elif moveTo == a1 or moveFrom == a1:
                newCastleState &= self.whiteCastleQueensideMask

            if moveTo == h8 or moveFrom == h8:
                newCastleState &= self.blackCastleKingsideMask
            elif moveTo == a8 or moveFrom == a8:
                newCastleState &= self.blackCastleQueensideMask

        # Update zobrist key with new piece position and side to move
        self.ZobristKey ^= int(Zobrist.sideToMove)
        self.ZobristKey ^= int(Zobrist.piecesArray[movePieceType, self.ColourToMoveIndex, moveFrom])
        self.ZobristKey ^= int(
            Zobrist.piecesArray[Piece.PieceType(pieceOnTargetSquare), self.ColourToMoveIndex, moveTo]
        )

        if oldEnPassantFile != 0:
            self.ZobristKey ^= int(Zobrist.enPassantFile[oldEnPassantFile])

        if newCastleState != originalCastleState:
            self.ZobristKey ^= Zobrist.castlingRights[originalCastleState]  # remove old castling rights state
            self.ZobristKey ^= Zobrist.castlingRights[newCastleState]  # add new castling rights state

        self.currentGameState |= newCastleState
        self.currentGameState |= self.fiftyMoveCounter << 14
        self.gameStateHistory.append(self.currentGameState)

        # Change side to move
        self.WhiteToMove = not self.WhiteToMove
        self.ColourToMove = Piece.White if self.WhiteToMove else Piece.Black
        self.OpponentColour = Piece.Black if self.WhiteToMove else Piece.White
        self.ColourToMoveIndex = 1 - self.ColourToMoveIndex
        self.plyCount += 1
        self.fiftyMoveCounter += 1

        if not inSearch:
            if movePieceType == Piece.Pawn or capturedPieceType != Piece.Empty:
                self.RepetitionPositionHistory = []
                self.fiftyMoveCounter = 0
            else:
                self.RepetitionPositionHistory.append(self.ZobristKey)

        return was_capture

    # Undo a move previously made on the board
    def UnmakeMove(self, move: Move, inSearch=False) -> None:
        # int opponentColour = ColourToMove
        self.opponentColourIndex = self.ColourToMoveIndex
        undoingWhiteMove = self.OpponentColour == Piece.White
        self.ColourToMove = self.OpponentColour  # side who made the move we are undoing
        self.OpponentColour = Piece.Black if undoingWhiteMove else Piece.White
        self.ColourToMoveIndex = 1 - self.ColourToMoveIndex
        self.WhiteToMove = not self.WhiteToMove

        originalCastleState = self.currentGameState & 0b1111

        capturedPieceType = (self.currentGameState >> 8) & 63
        capturedPiece = 0b0 if capturedPieceType == 0 else (capturedPieceType | self.OpponentColour)

        movedFrom = move.StartSquare()
        movedTo = move.TargetSquare()
        moveFlags = move.MoveFlag()
        isEnPassant = moveFlags == Move.Flag.EnPassantCapture
        isPromotion = move.IsPromotion()

        toSquarePieceType = Piece.PieceType(self.Square[movedTo])
        movedPieceType = Piece.Pawn if isPromotion else toSquarePieceType

        # Update zobrist key with new piece position and side to move
        self.ZobristKey ^= int(Zobrist.sideToMove)
        self.ZobristKey ^= int(
            Zobrist.piecesArray[movedPieceType, self.ColourToMoveIndex, movedFrom]
        )  # add piece back to square it moved from
        self.ZobristKey ^= int(
            Zobrist.piecesArray[toSquarePieceType, self.ColourToMoveIndex, movedTo]
        )  # remove piece from square it moved to

        oldEnPassantFile = (self.currentGameState >> 4) & 15
        if oldEnPassantFile != 0:
            self.ZobristKey ^= int(Zobrist.enPassantFile[oldEnPassantFile])

        # ignore ep captures, handled later
        if capturedPieceType != 0 and not isEnPassant:
            self.ZobristKey ^= int(Zobrist.piecesArray[capturedPieceType, self.opponentColourIndex, movedTo])
            self.GetPieceList(capturedPieceType, self.opponentColourIndex).AddPieceAtSquare(movedTo)

        # Update king index
        if movedPieceType == Piece.King:
            self.KingSquare[self.ColourToMoveIndex] = movedFrom
        elif not isPromotion:
            self.GetPieceList(movedPieceType, self.ColourToMoveIndex).MovePiece(movedTo, movedFrom)

        # put back moved piece
        self.Square[movedFrom] = (
            movedPieceType | self.ColourToMove
        )  # note that if move was a pawn promotion, this will put the promoted piece back instead of the pawn. Handled in special move switch
        self.Square[movedTo] = capturedPiece  # will be 0 if no piece was captured

        if isPromotion:
            self.pawns[self.ColourToMoveIndex].AddPieceAtSquare(movedFrom)
            match (moveFlags):
                case Move.Flag.PromoteToQueen:
                    self.queens[self.ColourToMoveIndex].RemovePieceAtSquare(movedTo)
                case Move.Flag.PromoteToKnight:
                    self.knights[self.ColourToMoveIndex].RemovePieceAtSquare(movedTo)
                case Move.Flag.PromoteToRook:
                    self.rooks[self.ColourToMoveIndex].RemovePieceAtSquare(movedTo)
                case Move.Flag.PromoteToBishop:
                    self.bishops[self.ColourToMoveIndex].RemovePieceAtSquare(movedTo)

        elif isEnPassant:  # ep cature: put captured pawn back on right square
            epIndex = movedTo + (-8 if self.ColourToMove == Piece.White else 8)
            self.Square[movedTo] = 0
            self.Square[epIndex] = capturedPiece
            self.pawns[self.opponentColourIndex].AddPieceAtSquare(epIndex)
            self.ZobristKey ^= int(Zobrist.piecesArray[Piece.Pawn, self.opponentColourIndex, epIndex])
        elif moveFlags == Move.Flag.Castling:  # castles: move rook back to starting square
            kingside = movedTo == 6 or movedTo == 62
            castlingRookFromIndex = movedTo + 1 if kingside else movedTo - 2
            castlingRookToIndex = movedTo - 1 if kingside else movedTo + 1

            self.Square[castlingRookToIndex] = 0
            self.Square[castlingRookFromIndex] = Piece.Rook | self.ColourToMove

            self.rooks[self.ColourToMoveIndex].MovePiece(castlingRookToIndex, castlingRookFromIndex)
            self.ZobristKey ^= int(Zobrist.piecesArray[Piece.Rook, self.ColourToMoveIndex, castlingRookFromIndex])
            self.ZobristKey ^= int(Zobrist.piecesArray[Piece.Rook, self.ColourToMoveIndex, castlingRookToIndex])

        self.gameStateHistory.pop()  # TODO removes current state from history
        self.currentGameState = self.gameStateHistory[-1]  # sets current state to previous state in history

        self.fiftyMoveCounter = (self.currentGameState & 4294950912) >> 14
        newEnPassantFile = (self.currentGameState >> 4) & 15
        if newEnPassantFile != 0:
            self.ZobristKey ^= int(Zobrist.enPassantFile[newEnPassantFile])

        newCastleState = self.currentGameState & 0b1111
        if newCastleState != originalCastleState:
            self.ZobristKey ^= Zobrist.castlingRights[originalCastleState]  # remove old castling rights state
            self.ZobristKey ^= Zobrist.castlingRights[newCastleState]  # add new castling rights state

        self.plyCount -= 1

        if not inSearch and len(self.RepetitionPositionHistory) > 0:
            self.RepetitionPositionHistory.pop()

    # Load custom position from fen string
    def LoadPosition(self, fen) -> None:
        loadedPosition = FenUtility.PositionFromFen(fen)
        # Load pieces into board array and piece lists
        for squareIndex in range(64):
            piece = loadedPosition.squares[squareIndex]
            self.Square[squareIndex] = piece

            if piece != Piece.Empty:
                pieceType = Piece.PieceType(piece)
                pieceColourIndex = WhiteIndex if Piece.IsColour(piece, Piece.White) else BlackIndex
                if Piece.IsSlidingPiece(piece):
                    if pieceType == Piece.Queen:
                        self.queens[pieceColourIndex].AddPieceAtSquare(squareIndex)
                    elif pieceType == Piece.Rook:
                        self.rooks[pieceColourIndex].AddPieceAtSquare(squareIndex)
                    elif pieceType == Piece.Bishop:
                        self.bishops[pieceColourIndex].AddPieceAtSquare(squareIndex)
                elif pieceType == Piece.Knight:
                    self.knights[pieceColourIndex].AddPieceAtSquare(squareIndex)
                elif pieceType == Piece.Pawn:
                    self.pawns[pieceColourIndex].AddPieceAtSquare(squareIndex)
                elif pieceType == Piece.King:
                    self.KingSquare[pieceColourIndex] = squareIndex

        # Side to move
        self.WhiteToMove = loadedPosition.whiteToMove
        self.ColourToMove = Piece.White if self.WhiteToMove else Piece.Black
        self.OpponentColour = Piece.Black if self.WhiteToMove else Piece.White
        self.ColourToMoveIndex = 0 if self.WhiteToMove else 1

        # Create gamestate
        whiteCastle = (1 << 0 if loadedPosition.whiteCastleKingside else 0) | (
            1 << 1 if loadedPosition.whiteCastleQueenside else 0
        )
        blackCastle = (1 << 2 if loadedPosition.blackCastleKingside else 0) | (
            1 << 3 if loadedPosition.blackCastleQueenside else 0
        )
        epState = loadedPosition.epFile << 4
        initialGameState = whiteCastle | blackCastle | epState
        self.gameStateHistory.append(initialGameState)
        self.currentGameState = initialGameState
        self.plyCount = loadedPosition.plyCount

        # Initialize zobrist key
        self.ZobristKey = Zobrist().CalculateZobristKey(self)
