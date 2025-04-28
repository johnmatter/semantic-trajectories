import sys
import os
import umap
import warnings
import mido # Assuming save_melody_as_midi uses mido
import numpy as np
import random

# Suppress specific warnings
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'", category=FutureWarning)
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state", category=UserWarning)

from .compressors import EmbeddingCompressor
from .memory_store import MemoryStore
from .utils import save_melody_as_midi

class TrajectoryGenerator:
    """Generates trajectories through a semantic memory space."""
    def __init__(self, store: MemoryStore):
        self.store = store
        self.compressor = store.compressor # Assumes compressor is accessible

    def generate(self, length: int = 10, start_id: int | None = None, strategy: str = 'random_walk_similar') -> list[int]:
        """
        Generates a trajectory (list of memory IDs).

        Args:
            length: The desired length of the trajectory.
            start_id: Optional starting memory ID. If None, chooses randomly.
            strategy: The method for choosing the next step ('random_walk_similar').

        Returns:
            A list of memory IDs representing the trajectory.
        """
        if not self.store.memory:
            return []

        memory_ids = list(self.store.memory.keys())
        if start_id is None or start_id not in memory_ids:
            current_id = random.choice(memory_ids)
        else:
            current_id = start_id

        trajectory = [current_id]

        if strategy == 'random_walk_similar':
            for _ in range(length - 1):
                candidates = [(id, self.compressor.similarity(self.store.memory[current_id], self.store.memory[id]))
                              for id in memory_ids if id != current_id]
                if not candidates:
                    break
                # Sort by similarity (descending)
                candidates.sort(key=lambda x: -x[1])
                # Pick next step from top 3 most similar (could be parameterized)
                next_id, _ = random.choice(candidates[:min(3, len(candidates))])
                trajectory.append(next_id)
                current_id = next_id
        else:
            # Future strategies could be added here (e.g., purely random, furthest neighbor)
            raise NotImplementedError(f"Strategy '{strategy}' not implemented.")

        return trajectory

class MelodyMapper:
    """Maps a semantic trajectory to a musical melody."""

    def map_trajectory(self, trajectory: list[int], store: MemoryStore, method: str = 'bias') -> list[tuple[int, int]]:
        """
        Maps a trajectory of memory IDs to a sequence of MIDI notes.

        Args:
            trajectory: A list of memory IDs.
            store: The MemoryStore containing the corresponding vectors.
            method: The mapping method ('basic' or 'bias').

        Returns:
            A list of (pitch, duration) tuples.
        """
        notes = []
        if not trajectory:
            return notes

        
        if len(trajectory) < 2: return [(60, 480)] # Need at least two points for bias method

        # --- UMAP Projection ---
        # Get only the vectors present in the trajectory
        trajectory_vectors = [store.memory[mem_id] for mem_id in trajectory]
        memory_array = np.array(trajectory_vectors)

          # Check if enough unique samples for UMAP
        unique_vecs = np.unique(memory_array, axis=0)
        if len(unique_vecs) < 2:
              print("Warning: Not enough unique vectors in trajectory for UMAP, returning simple note.")
              return [(60, 480)] # Fallback if not enough unique points

        # Explicitly set n_neighbors
        n_neighbors = min(15, unique_vecs.shape[0] - 1)
        if n_neighbors < 2: n_neighbors = 2 # UMAP requires n_neighbors >= 2

        try:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
            embedding = reducer.fit_transform(unique_vecs) # Fit on unique vectors
              # Create mapping from original trajectory index to its embedded point
              # This handles duplicate vectors in the trajectory
            vec_to_point = {tuple(vec): point for vec, point in zip(unique_vecs, embedding)}
            trajectory_points = [vec_to_point[tuple(vec)] for vec in trajectory_vectors]

        except Exception as e:
              print(f"Error during UMAP embedding: {e}. Returning simple note.")
              return [(60, 480)] # Fallback on UMAP error


        # --- Map Points to Notes ---
        last_point = None
        last_pitch = 60  # Start at middle C

        for point in trajectory_points:
            if last_point is None:
                pitch = last_pitch
                duration = 480 # Default duration for the first note
            else:
                delta = point - last_point
                dy = delta[1]  # Vertical movement
                dx = delta[0]  # Horizontal movement

                # Map dy to pitch shift (adjust scaling factor as needed)
                pitch_shift = int(dy * 5)
                pitch = max(0, min(127, last_pitch + pitch_shift))

                # Map dx to rhythm modification (prevent division by zero or negative durations)
                # Larger dx -> smaller rhythm_modifier -> longer duration
                # Smaller dx -> larger rhythm_modifier -> shorter duration
                # Let's refine this: perhaps map absolute dx to deviation from base duration?
                # Or use dx to control articulation/velocity later?
                # Simpler approach: Use dx to slightly modify base duration
                rhythm_modifier = np.clip(1 + dx, 0.5, 2.0) # Clamp modifier
                base_duration = 480
                duration = int(base_duration / rhythm_modifier)
                # Clamp duration to reasonable range
                duration = max(60, min(960, duration))

            notes.append((pitch, duration))
            last_point = point
            last_pitch = pitch

        return notes

class MidiGenerator:
    """Orchestrates the generation of MIDI from semantic memories."""
    def __init__(self, memories: list[str], mapper: MelodyMapper):
        self.compressor = EmbeddingCompressor()
        self.store = MemoryStore(self.compressor)
        self.trajectory_gen = TrajectoryGenerator(self.store)
        self.mapper = mapper

        # Add initial memories
        for mem in memories:
            self.store.add(mem)

    def generate_midi(self, output_filename: str, trajectory_length: int = 10, mapping_method: str = 'bias'):
        """
        Generates a trajectory, maps it to notes, and saves a MIDI file.

        Args:
            output_filename: Path to save the MIDI file.
            trajectory_length: The length of the semantic trajectory to generate.
            mapping_method: The method used by the mapper (e.g., 'bias', 'basic').
        """
        if not self.store.memory:
            print("Error: Memory store is empty. Cannot generate MIDI.")
            return

        print(f"Generating trajectory (length {trajectory_length})...")
        trajectory = self.trajectory_gen.generate(length=trajectory_length)
        print(f"Trajectory: {trajectory}")

        if not trajectory:
            print("Error: Failed to generate trajectory.")
            return

        print(f"Mapping trajectory to notes using method '{mapping_method}'...")
        notes = self.mapper.map_trajectory(trajectory, self.store, method=mapping_method)
        print(f"Generated {len(notes)} notes.")
        # print(f"Notes: {notes}") # Optional: print notes for debugging

        if not notes:
            print("Error: Failed to map trajectory to notes.")
            return

        print(f"Saving MIDI file to {output_filename}...")
        try:
            # Assuming save_melody_as_midi handles potential errors
            save_melody_as_midi(notes, filename=output_filename)
            print("Saved successfully!")
        except Exception as e:
            print(f"Error saving MIDI file: {e}")
