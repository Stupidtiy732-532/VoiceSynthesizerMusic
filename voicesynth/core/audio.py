
import numpy as np
import soundfile as sf


class AudioBuffer:

    def __init__(
        self,
        samples,
        sample_rate=44100
    ):

        self.samples = np.asarray(
            samples,
            dtype=np.float32
        )

        self.sample_rate = sample_rate

    @property
    def duration(self):

        return len(self.samples) / self.sample_rate

    def save(self, path):

        sf.write(
            path,
            self.samples,
            self.sample_rate
        )

    def append(self, other):

        if self.sample_rate != other.sample_rate:

            raise ValueError(
                "Sample rates must match"
            )

        self.samples = np.concatenate(
            (self.samples, other.samples)
        )

    def mix(self, other, gain=1.0):

        if self.sample_rate != other.sample_rate:

            raise ValueError(
                "Sample rates must match"
            )

        length = max(
            len(self.samples),
            len(other.samples)
        )

        result = np.zeros(
            length,
            dtype=np.float32
        )

        result[:len(self.samples)] += self.samples

        result[:len(other.samples)] += (
            other.samples * gain
        )

        return AudioBuffer(
            result,
            self.sample_rate
        )