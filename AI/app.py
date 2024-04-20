from Board import Board
from FenUtility import FenUtility
from MoveGenerator import MoveGenerator
import time
from Search import Search, AISettings, SearchDiagnostics
# from chess import Board as ChessBoard

from flask import Flask, request, jsonify

app = Flask(__name__)

settings = AISettings()
settings.useIterativeDeepening = True
settings.useTranspositionTable = True
settings.useFixedDepthSearch = True
settings.endlessSearchMode = False
settings.clearTTEachMove = True
settings.useBook = False
settings.maxBookPly = 10
settings.searchTimeMillis = 10000
settings.promotionsToSearch = MoveGenerator.PromotionMode.All
settings.depth = 4
# check only moves for last captured TODO
settings.diagnostics = SearchDiagnostics()
settings.only_square = False

@app.route("/", methods=['POST'])
def hello():
    data = request.get_json()
    fen = data.get('fen')

    try:
        board = Board()
        board.LoadPosition(fen)
        search = Search(board, settings)
        move, eval = search.StartSearch()
    except:
        return "", 500

    return jsonify({"move": move.Name()}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888)


# bb = ChessBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
# search2 = Search(board, settings2)
# while True:
#     move = input("move:" )
#     bb.push_uci(move)
#     fen = bb.fen()
#     board = Board()
#     board.LoadPosition(fen)
#
#     search = Search(board, settings)
#     start = time.time()
#     move, eval = search.StartSearch()
#     print(time.time() - start)
#     print(move.Name(), eval)
#     search.board.MakeMove(move)
#     fen = FenUtility.CurrentFen(search.board)
#     print(fen)
#     bb.set_fen(fen)
    # print(FenUtility.CurrentFen(search.board))
    # exit()
    #
    # fen = FenUtility.CurrentFen(board)
    # s.set_fen_position(fen)
    # print("============================")
