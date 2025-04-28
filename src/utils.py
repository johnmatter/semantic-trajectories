import numpy as np
import mido

def normalize(vec):
    return vec / np.linalg.norm(vec)

def vector_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def save_melody_as_midi(notes, filename="output.mid", tempo=500000):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    for pitch, duration in notes:
        track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
        track.append(mido.Message('note_off', note=pitch, velocity=64, time=int(duration)))

    mid.save(filename)
