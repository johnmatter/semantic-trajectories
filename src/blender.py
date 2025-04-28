import numpy as np
import random

class MemoryBlender:
    def __init__(self, store, compressor):
        self.store = store
        self.compressor = compressor

    def blend_random(self, num_sources=2):
        """
        Randomly blend `num_sources` memories into a new vector.
        """
        ids = random.sample(list(self.store.memory.keys()), num_sources)
        vectors = [self.store.memory[id] for id in ids]

        blended = np.mean(vectors, axis=0)
        blended = blended / np.linalg.norm(blended)  # normalize
        return blended

    def blend_nearby(self, current_id, num_sources=2, top_k=5):
        """
        Blend with similar memories to the current one.
        """
        candidates = [(id, self.compressor.similarity(
            self.store.memory[current_id], self.store.memory[id]))
                      for id in self.store.memory if id != current_id]
        candidates.sort(key=lambda x: -x[1])
        top_candidates = [id for id, _ in candidates[:top_k]]

        chosen = random.sample(top_candidates, min(num_sources, len(top_candidates)))
        vectors = [self.store.memory[id] for id in chosen]

        blended = np.mean(vectors, axis=0)
        blended = blended / np.linalg.norm(blended)
        return blended

    def mutate(vec, noise_level=0.05):
        noise = np.random.randn(*vec.shape) * noise_level
        mutated = vec + noise
        return mutated / np.linalg.norm(mutated)

# e.g.:
# blender = MemoryBlender(store, compressor)
# trajectory = []
# last_vec = None
# current_id = random.choice(memory_ids)

# for _ in range(10):
#     if random.random() < 0.3:  # 30% chance to blend
#         vec = blender.blend_nearby(current_id)
#     else:
#         vec = store.memory[current_id]

#     trajectory.append(vec)

#     # Find next based on similarity
#     candidates = [(id, compressor.similarity(vec, store.memory[id]))
#                   for id in memory_ids if id != current_id]
#     if not candidates:
#         break
#     candidates.sort(key=lambda x: -x[1])
#     next_id, _ = random.choice(candidates[:3])
#     current_id = next_id
