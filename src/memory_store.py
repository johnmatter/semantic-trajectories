import pickle

class MemoryStore:
    def __init__(self, compressor):
        self.compressor = compressor
        self.memory = {}  # id -> compressed object
        self.raw_memory = {}  # id -> original text
        self.next_id = 0

    def add(self, text: str):
        compressed = self.compressor.compress(text)
        self.memory[self.next_id] = compressed
        self.raw_memory[self.next_id] = text
        self.next_id += 1

    def find_similar(self, text: str, threshold=0.5):
        if not hasattr(self.compressor, "similarity"):
            raise NotImplementedError("Compressor does not support similarity search.")

        query_compressed = self.compressor.compress(text)
        similar_ids = []

        for id, stored_compressed in self.memory.items():
            score = self.compressor.similarity(query_compressed, stored_compressed)
            if score >= threshold:
                similar_ids.append((id, score))

        similar_ids.sort(key=lambda x: x[1], reverse=True)
        return similar_ids

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.memory, self.raw_memory, self.next_id), f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.memory, self.raw_memory, self.next_id = pickle.load(f)
