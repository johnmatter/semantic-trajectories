from abc import ABC, abstractmethod
from datasketch import MinHash
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCompressor(ABC):
    @abstractmethod
    def compress(self, text: str):
        pass

    @abstractmethod
    def expand(self, compressed):
        pass

class TokenCompressor(SemanticCompressor):
    def compress(self, text: str):
        return text.split()

    def expand(self, compressed):
        return " ".join(compressed)

class SemanticHashCompressor(SemanticCompressor):
    def __init__(self, num_perm=128):
        self.num_perm = num_perm

    def compress(self, text: str):
        tokens = text.lower().split()
        m = MinHash(num_perm=self.num_perm)
        for token in tokens:
            m.update(token.encode('utf8'))
        return m

    def expand(self, compressed):
        raise NotImplementedError("Semantic hashes are one-way.")

    def similarity(self, hash1, hash2):
        return hash1.jaccard(hash2)

class EmbeddingCompressor(SemanticCompressor):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def compress(self, text: str):
        emb = self.model.encode(text)
        return emb

    def expand(self, compressed):
        raise NotImplementedError("Embeddings are one-way.")

    def similarity(self, emb1, emb2):
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return np.dot(emb1, emb2)
