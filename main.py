
import os

from voicesynth.core.score import Note, Score
from voicesynth.engines.classical import ClassicalEngine


def main():

    score = Score(tempo=120)

    melody = [

        ("a", 60, 0.8),
        ("s", 60, 0.8),
        ("f", 60, 0.8),
        ("m", 60, 0.8),
        ("l", 60, 0.8),
        ("ka", 60, 0.8),

    ]

    for phoneme, midi_note, duration in melody:

        score.add_note(
            Note(
                phoneme=phoneme,
                midi_note=midi_note,
                duration=duration
            )
        )

    engine = ClassicalEngine()

    audio = engine.synthesize(score)

    os.makedirs(
        "output",
        exist_ok=True
    )

    audio.save(
        "output/singing.wav"
    )

    print(
        "Generated:",
        audio.duration,
        "seconds"
    )

    print(
        "Saved to:",
        "output/singing.wav"
    )


if __name__ == "__main__":

    main()