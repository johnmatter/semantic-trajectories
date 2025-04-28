from src.trajectory_generator import MelodyMapper, MidiGenerator

# Define the initial memories
initial_memories = [
    "John Matter is an electroacoustic composer.",
    "He works with Max/MSP and unconventional technology pairings.",
    "Music can emerge from structures.",
    "Semantic destruction is a creative act.",
    "Topology can inform melodic contours.",
    "The unconcsious is structure like a language."
    "The mind moves through memory like a melody.",
    "The non-duped err."
]

generator = MidiGenerator(memories=initial_memories, mapper=MelodyMapper())
generator.generate_midi("output_bias.mid", trajectory_length=15, mapping_method='bias')
