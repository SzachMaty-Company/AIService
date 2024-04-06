from Piece import Piece
from BoardRepresentation import *
from const import *


class LoadedPositionInfo:
    squares: list[int]
    whiteCastleKingside: bool
    whiteCastleQueenside: bool
    blackCastleKingside: bool
    blackCastleQueenside: bool
    epFile = 0
    whiteToMove: bool
    plyCount = 0


class FenUtility:
    pieceTypeFromSymbol = {
        "k": Piece.King,
        "p": Piece.Pawn,
        "n": Piece.Knight,
        "b": Piece.Bishop,
        "r": Piece.Rook,
        "q": Piece.Queen,
    }


    # Load position from fen string
    @staticmethod
    def PositionFromFen(fen) -> LoadedPositionInfo:
        loadedPositionInfo = LoadedPositionInfo()
        loadedPositionInfo.squares = [0] * 64
        sections = fen.split(" ")

        file = 0
        rank = 7

        for symbol in sections[0]:
            if symbol == "/":
                file = 0
                rank -= 1
            else:
                if str.isdigit(symbol):
                    file += int(symbol)
                else:
                    pieceColour = Piece.White if str.isupper(symbol) else Piece.Black
                    pieceType = FenUtility.pieceTypeFromSymbol[str.lower(symbol)]
                    loadedPositionInfo.squares[rank * 8 + file] = pieceType | pieceColour
                    file += 1

        loadedPositionInfo.whiteToMove = sections[1] == "w"

        castlingRights = sections[2] if len(sections) > 2 else "KQkq"
        loadedPositionInfo.whiteCastleKingside = "K" in castlingRights
        loadedPositionInfo.whiteCastleQueenside = "Q" in castlingRights
        loadedPositionInfo.blackCastleKingside = "k" in castlingRights
        loadedPositionInfo.blackCastleQueenside = "q" in castlingRights

        if len(sections) > 3:
            enPassantFileName = str(sections[3][0])
            if enPassantFileName in fileNames:
                loadedPositionInfo.epFile = fileNames.index(enPassantFileName) + 1
        # Half-move clock
        if len(sections) > 4:
            loadedPositionInfo.plyCount = int(sections[4])

        return loadedPositionInfo

    # Get the fen string of the current position
    @staticmethod
    def CurrentFen(board) -> str:
        fen = ""
        for rank in range(7, -1, -1):
            numEmptyFiles = 0
            for file in range(8):
                i = rank * 8 + file
                piece = board.Square[i]
                if piece != 0:
                    if numEmptyFiles != 0:
                        fen += str(numEmptyFiles)
                        numEmptyFiles = 0
                    isBlack = Piece.IsColour(piece, Piece.Black)
                    pieceType = Piece.PieceType(piece)
                    pieceChar = " "
                    match (pieceType):
                        case Piece.Rook:
                            pieceChar = "R"
                        case Piece.Knight:
                            pieceChar = "N"
                        case Piece.Bishop:
                            pieceChar = "B"
                        case Piece.Queen:
                            pieceChar = "Q"
                        case Piece.King:
                            pieceChar = "K"
                        case Piece.Pawn:
                            pieceChar = "P"
                    fen += pieceChar.lower() if isBlack else pieceChar
                else:
                    numEmptyFiles += 1

            if numEmptyFiles != 0:
                fen += str(numEmptyFiles)

            if rank != 0:
                fen += "/"

        # Side to move
        fen += " "
        fen += "w" if board.WhiteToMove else "b"

        # Castling
        whiteKingside = (board.currentGameState & 1) == 1
        whiteQueenside = (board.currentGameState >> 1 & 1) == 1
        blackKingside = (board.currentGameState >> 2 & 1) == 1
        blackQueenside = (board.currentGameState >> 3 & 1) == 1
        fen += " "
        fen += "K" if whiteKingside else ""
        fen += "Q" if whiteQueenside else ""
        fen += "k" if blackKingside else ""
        fen += "q" if blackQueenside else ""
        fen += "-" if ((board.currentGameState & 15) == 0) else ""

        # En-passant
        fen += " "
        epFile = (board.currentGameState >> 4) & 15
        if epFile == 0:
            fen += "-"
        else:
            fileName = fileNames[epFile - 1]
            epRank = 6 if board.WhiteToMove else 3
            fen += fileName + str(epRank)

        # 50 move counter
        fen += " "
        fen += str(board.fiftyMoveCounter)

        # Full-move count (should be one at start, and increase after each move by black)
        fen += " "
        fen += str(int((board.plyCount / 2) + 1))

        return fen
