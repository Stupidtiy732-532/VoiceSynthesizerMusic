
class Note:

    def __init__(
        self,
        phoneme,
        midi_note,
        duration,
        velocity=1.0,
        pitch_curve=None
    ):

        self.phoneme = phoneme
        self.midi_note = midi_note
        self.duration = duration
        self.velocity = velocity
        self.pitch_curve = pitch_curve

    @property
    def frequency(self):

        return 440.0 * 2 ** (
            (self.midi_note - 69) / 12
        )


class Score:

    def __init__(self, tempo=120):

        self.tempo = tempo
        self.notes = []

    def add_note(self, note):

        self.notes.append(note)

    @property
    def duration(self):

        return sum(
            note.duration
            for note in self.notes
        )