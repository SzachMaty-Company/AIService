class RepetitionTable:
        hashes: list[int]
        startIndices: list[int]
        count: int

        def __init__(self):
            self.hashes = [0] * 128
            self.startIndices = [0] * 128

        def Init(self, initialHashes: list[int]) -> None:
            self.count = len(initialHashes)
            for i in range(len(initialHashes)):
                self.hashes[i] = initialHashes[i]
                self.startIndices[i] = 0
            self.startIndices[self.count] = 0

        def Push(self, hash, reset) -> None:
            self.hashes[self.count] = hash
            self.count += 1
            self.startIndices[self.count] =  self.count - 1 if reset else self.startIndices[self.count - 1]

        def TryPop(self) -> None:
            self.count =max(0, self.count - 1)

        def Contains(self, h) -> bool:
            for i in range(self.startIndices[self.count], self.count):
                if self.hashes[i] == h:
                    return True
