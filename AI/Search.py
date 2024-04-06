import math
import time
from RepetitionTable import RepetitionTable
from Move import Move
from TranspositionTable import TranspositionTable
from MoveGenerator import MoveGenerator
from Board import Board
from Evaluation import Evaluation
from MoveOrdering import MoveOrdering
from FenUtility import FenUtility
from Piece import  Piece

class SearchDiagnostics:
    lastCompletedDepth: int
    moveVal: str
    move: str
    eval: int
    isBook: bool
    numPositionsEvaluated = 0


class AISettings:
    depth: int
    useIterativeDeepening: bool
    useTranspositionTable: bool

    useFixedDepthSearch: bool
    searchTimeMillis = 10000
    endlessSearchMode: bool
    clearTTEachMove: bool

    only_square: bool

    useBook: bool
    # book: TextAsset
    maxBookPly = 10

    promotionsToSearch: int

    diagnostics: SearchDiagnostics

    # def RequestAbortSearch () -> None:
    #     requestAbortSearch?.Invoke ()


class Search:
    transpositionTableSize = 64000
    immediateMateScore = 100000
    positiveInfinity = 9999999
    negativeInfinity = -positiveInfinity

    # public event System.Action<Move> onSearchComplete

    tt: TranspositionTable
    moveGenerator: MoveGenerator

    bestMoveThisIteration: Move
    bestEvalThisIteration: int
    bestMove: Move
    bestEval: int
    currentIterativeSearchDepth: int
    abortSearch: bool

    invalidMove: Move
    moveOrdering: MoveOrdering
    settings: AISettings
    board: Board
    evaluation: Evaluation
    repetitionTable: RepetitionTable

    # Diagnostics
    searchDiagnostics: SearchDiagnostics
    numNodes = 0
    numQNodes = 0
    numCutoffs = 0
    numTranspositions = 0
    start_time: time.time
    # System.Diagnostics.Stopwatch searchStopwatch

    def __init__(self, board: Board, settings: AISettings):
        self.board = board
        self.settings = settings
        self.evaluation = Evaluation()
        self.moveGenerator = MoveGenerator()
        self.tt = TranspositionTable(self.board, self.transpositionTableSize)
        self.moveOrdering = MoveOrdering(self.moveGenerator, self.tt)
        self.invalidMove = Move.InvalidMove()
        self.repetitionTable = RepetitionTable()
        # s = TranspositionTable.Entry.GetSize ()
        # Debug.Log ("TT entry: " + s + " bytes. Total size: " + ((s * transpositionTableSize) / 1000f) + " mb.")

    def StartSearch(self) -> tuple[Move, int]:
        # InitDebugInfo ()

        # Initialize search settings
        self.bestEvalThisIteration = bestEval = 0
        self.bestMoveThisIteration = Move.InvalidMove()
        self.bestMove = Move.InvalidMove()
        self.tt.enabled = self.settings.useTranspositionTable

        # Clearing the transposition table before each search seems to help
        # This makes no sense to me, I presume there is a bug somewhere but haven't been able to track it down yet
        if self.settings.clearTTEachMove:
            self.tt.Clear()
        self.repetitionTable.Init(self.board.RepetitionPositionHistory)
        self.moveGenerator.promotionsToGenerate = self.settings.promotionsToSearch
        self.currentIterativeSearchDepth = 0
        self.abortSearch = False
        self.searchDiagnostics = SearchDiagnostics()

        # Iterative deepening. This means doing a full search with a depth of 1, then with a depth of 2, and so on.
        # This allows the search to be aborted at any time, while still yielding a useful result from the last search.
        self.start_time = time.time()
        if self.settings.useIterativeDeepening:
            targetDepth = self.settings.depth if self.settings.useFixedDepthSearch else 99999

            for searchDepth in range(1, targetDepth):
                self.SearchMoves(searchDepth, 0, self.negativeInfinity, self.positiveInfinity)
                if self.abortSearch:
                    break
                else:
                    self.currentIterativeSearchDepth = searchDepth
                    self.bestMove = self.bestMoveThisIteration
                    self.bestEval = self.bestEvalThisIteration

                    # Update diagnostics
                    self.searchDiagnostics.lastCompletedDepth = searchDepth
                    self.searchDiagnostics.move = self.bestMove.Name
                    self.searchDiagnostics.eval = self.bestEval
                    self.searchDiagnostics.moveVal = str(FenUtility.CurrentFen(self.board) + "," + self.bestMove.Name())

                    # Exit search if found a mate
                    if self.IsMateScore(bestEval) and not self.settings.endlessSearchMode:
                        break
        else:
            self.SearchMoves(self.settings.depth, 0, self.negativeInfinity, self.positiveInfinity)
            self.bestMove = self.bestMoveThisIteration
            self.bestEval = self.bestEvalThisIteration

        return self.bestMove, self.bestEval
        # onSearchComplete?.Invoke (bestMove) TODO

    def GetSearchResult(self) -> tuple[Move, int]:
        return (self.bestMove, self.bestEval)

    def EndSearch(self) -> None:
        self.abortSearch = True

    def SearchMoves(self, depth, plyFromRoot, alpha, beta, was_capture=False, target_square=None) -> int:
        if time.time() - self.start_time > self.settings.searchTimeMillis / 1000:
            self.abortSearch = True
        if self.abortSearch:
            return 0

        if plyFromRoot > 0 or self.repetitionTable.Contains(self.board.currentGameState):
            # Detect draw by repetition.
            # Returns a draw score even if this position has only appeared once in the game history (for simplicity).
            if self.board.ZobristKey in self.board.RepetitionPositionHistory:
                return 0

            # Skip this position if a mating sequence has already been found earlier in
            # the search, which would be shorter than any mate we could find from here.
            # This is done by observing that alpha can't possibly be worse (and likewise
            # beta can't  possibly be better) than being mated in the current position.
            alpha = max(alpha, -self.immediateMateScore + plyFromRoot)
            beta = min(beta, self.immediateMateScore - plyFromRoot)
            if alpha >= beta:
                return alpha

        # Try looking up the current position in the transposition table.
        # If the same position has already been searched to at least an equal depth
        # to the search we're doing now,we can just use the recorded evaluation.
        ttVal = self.tt.LookupEvaluation(depth, plyFromRoot, alpha, beta)
        if ttVal != TranspositionTable.lookupFailed:
            self.numTranspositions += 1
            if plyFromRoot == 0:
                self.bestMoveThisIteration = self.tt.GetStoredMove()
                self.bestEvalThisIteration = self.tt.entries[self.tt.Index()].value
                # Debug.Log ("move retrieved " + bestMoveThisIteration.Name + " Node type: " + tt.entries[tt.Index].nodeType + " depth: " + tt.entries[tt.Index].depth)

            return ttVal

        if depth == 0:
            return self.QuiescenceSearch(alpha, beta, 5, target_square)

        moves: list[Move] = self.moveGenerator.GenerateMoves(self.board)
        moves = self.moveOrdering.OrderMoves(self.board, moves, self.settings.useTranspositionTable)
        # Detect checkmate and stalemate when no legal moves are available
        if len(moves) == 0:
            if self.moveGenerator.inCheck:
                mateScore = self.immediateMateScore - plyFromRoot
                return -mateScore
            else:
                return 0
        if (plyFromRoot > 0):
            wasPawnMove = Piece.PieceType(self.board.Square[target_square]) == Piece.Pawn
            self.repetitionTable.Push(self.board.currentGameState, was_capture or wasPawnMove)

        evalType = TranspositionTable.UpperBound
        bestMoveInThisPosition: Move = self.invalidMove
        for i in range(len(moves)):
            was_capture = self.board.MakeMove(moves[i], inSearch=True)
            eval = -self.SearchMoves(depth - 1, plyFromRoot + 1, -beta, -alpha, was_capture, moves[i].TargetSquare())
            self.board.UnmakeMove(moves[i], inSearch=True)
            self.numNodes += 1

            # Move was *too* good, so opponent won't allow this position to be reached
            # (by choosing a different move earlier on). Skip remaining moves.
            if eval >= beta:
                self.tt.StoreEvaluation(depth, plyFromRoot, beta, TranspositionTable.LowerBound, moves[i])
                self.numCutoffs += 1
                return beta

            # Found a new best move in this position
            if eval > alpha:
                evalType = TranspositionTable.Exact
                bestMoveInThisPosition = moves[i]

                alpha = eval
                if plyFromRoot == 0:
                    self.bestMoveThisIteration = moves[i]
                    self.bestEvalThisIteration = eval
                if plyFromRoot > 0:
                    self.repetitionTable.TryPop()

        if plyFromRoot > 0:
            self.repetitionTable.TryPop()

        self.tt.StoreEvaluation(depth, plyFromRoot, alpha, evalType, bestMoveInThisPosition)

        return alpha

    # Search capture moves until a 'quiet' position is reached.
    def QuiescenceSearch(self, alpha, beta, depth, target_square) -> int:
        # A player isn't forced to make a capture (typically), so see what the evaluation is without capturing anything.
        # This prevents situations where a player ony has bad captures available from being evaluated as bad,
        # when the player might have good non-capture moves available.
        eval = self.evaluation.Evaluate(self.board)
        # if depth == 0:
        #     return eval
        self.searchDiagnostics.numPositionsEvaluated += 1
        if eval >= beta:
            return beta

        if eval > alpha:
            alpha = eval

        moves = self.moveGenerator.GenerateMoves(self.board, False)
        moves = self.moveOrdering.OrderMoves(self.board, moves, False)
        for i in range(len(moves)):
            self.board.MakeMove(moves[i], True)
            eval = -self.QuiescenceSearch(-beta, -alpha, depth-1, target_square)
            self.board.UnmakeMove(moves[i], True)
            self.numQNodes += 1

            if eval >= beta:
                self.numCutoffs += 1
                return beta

            if eval > alpha:
                alpha = eval

        return alpha

    @classmethod
    def IsMateScore(cls, score) -> bool:
        maxMateDepth = 1000
        return abs(score) > cls.immediateMateScore - maxMateDepth

    @classmethod
    def NumPlyToMateFromScore(cls, score) -> int:
        return cls.immediateMateScore - abs(score)

    def LogDebugInfo(self) -> None:
        self.AnnounceMate()
        print(
            f"Best move: {self.bestMoveThisIteration.Name} Eval: {self.bestEvalThisIteration} Search time: {self.searchStopwatch.ElapsedMilliseconds} ms."
        )
        print(
            f"Num nodes: {self.numNodes} num Qnodes: {self.numQNodes} num cutoffs: {self.numCutoffs} num TThits {self.numTranspositions}"
        )

    def AnnounceMate(self) -> None:
        if self.IsMateScore(self.bestEvalThisIteration):
            numPlyToMate = self.NumPlyToMateFromScore(self.bestEvalThisIteration)
            # int numPlyToMateAfterThisMove = numPlyToMate - 1

            numMovesToMate = int(math.ceil(numPlyToMate / 2))

            sideWithMate = (
                "Black" if (self.bestEvalThisIteration * (1 if self.board.WhiteToMove else -1) < 0) else "White"
            )

            print(f"{sideWithMate} can mate in {numMovesToMate} moves")
