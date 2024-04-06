from Board import Board
from Move import Move


class TranspositionTable:
    class Entry:
        key: int
        value: int
        move: Move
        depth: bytes
        nodeType: bytes

        # 	public readonly byte gamePly

        def __init__(self, key, value, depth, nodeType, move: Move):
            self.key = key
            self.value = value
            self.depth = depth  # depth is how many ply were searched ahead from this position
            self.nodeType = nodeType
            self.move = move

    lookupFailed = -999999

    # The value for this position is the exact evaluation
    Exact = 0
    # A move was found during the search that was too good, meaning the opponent will play a different move earlier on,
    # not allowing the position where this move was available to be reached. Because the search cuts off at
    # this point (beta cut-off), an even better move may exist. This means that the evaluation for the
    # position could be even higher, making the stored value the lower bound of the actual value.
    LowerBound = 1
    # No move during the search resulted in a position that was better than the current player could get from playing a
    # different move in an earlier position (i.e eval was <= alpha for all moves in the position).
    # Due to the way alpha-beta search works, the value we get here won't be the exact evaluation of the position,
    # but rather the upper bound of the evaluation. This means that the evaluation is, at most, equal to this value.
    UpperBound = 2

    entries: list[Entry| None]

    size: int
    enabled = True
    board: Board

    def __init__(self, board: Board, size):
        self.board = board
        self.size = size

        self.entries = [None] * self.size

    def Clear(self) -> None:
        self.entries = [None] * self.size
        # for i in range(len(self.entries)):
        #     self.entries[i] = []

    def Index(self) -> int:
        return self.board.ZobristKey % self.size

    def GetStoredMove(self) -> Move:
        x = self.entries[self.Index()]
        if x is not None:
            return x.move
        return Move.InvalidMove()

    def LookupEvaluation(self, depth, plyFromRoot, alpha, beta) -> int:
        if not self.enabled:
            return self.lookupFailed
        entry = self.entries[self.Index()]

        if entry is not None and entry.key == self.board.ZobristKey:
            # Only use stored evaluation if it has been searched to at least the same depth as would be searched now
            if entry.depth >= depth:
                correctedScore = self.CorrectRetrievedMateScore(entry.value, plyFromRoot)
                # We have stored the exact evaluation for this position, so return it
                if entry.nodeType == self.Exact:
                    return correctedScore
                # We have stored the upper bound of the eval for this position. If it's less than alpha then we don't need to
                # search the moves in this position as they won't interest us otherwise we will have to search to find the exact value
                if entry.nodeType == self.UpperBound and correctedScore <= alpha:
                    return correctedScore
                # We have stored the lower bound of the eval for this position. Only return if it causes a beta cut-off.
                if entry.nodeType == self.LowerBound and correctedScore >= beta:
                    return correctedScore

        return self.lookupFailed

    def StoreEvaluation(self, depth, numPlySearched, eval, evalType, move: Move) -> None:
        if not self.enabled:
            return
        # ulong index = Index
        # if (depth >= entries[Index].depth) {
        entry = self.Entry(
            self.board.ZobristKey,
            self.CorrectMateScoreForStorage(eval, numPlySearched),
            depth,
            evalType,
            move,
        )
        self.entries[self.Index()] = entry
        # }

    def CorrectMateScoreForStorage(self, score, numPlySearched) -> int:
        from Search import Search
        if Search.IsMateScore(score):
            sign = -1 if score < 0 else 1
            return (score * sign + numPlySearched) * sign

        return score

    def CorrectRetrievedMateScore(self, score, numPlySearched) -> int:
        from Search import Search
        if Search.IsMateScore(score):
            sign = -1 if score < 0 else 1
            return (score * sign - numPlySearched) * sign

        return score
