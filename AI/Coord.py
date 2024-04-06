class Coord:
    fileIndex: int
    rankIndex: int

    def __init__(self, fileIndex, rankIndex):
        self.fileIndex = fileIndex
        self.rankIndex = rankIndex

    def IsLightSquare(self) -> bool:
        return (self.fileIndex + self.rankIndex) % 2 != 0

    def __eq__(self, other):
        return self.fileIndex == other.file_index and self.rankIndex == other.rank_index
