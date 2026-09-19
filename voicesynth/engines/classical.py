import numpy as np
from scipy.signal import iirpeak, lfilter

from voicesynth.core.audio import AudioBuffer
from voicesynth.core.engine import SynthEngine


class ClassicalEngine(SynthEngine):

    SAMPLE_RATE = 44100

    VIBRATO_RATE = 5.5
    VIBRATO_DEPTH = 0.012

    HARMONICS = 35

    FORMANTS = {

        "a": (
            (800, 100),
            (1150, 120),
            (2900, 180),
        ),

        "e": (
            (500, 80),
            (1900, 120),
            (2600, 180),
        ),

        "i": (
            (300, 70),
            (2200, 120),
            (3000, 180),
        ),

        "o": (
            (450, 80),
            (800, 100),
            (2830, 180),
        ),

        "u": (
            (350, 70),
            (900, 100),
            (2200, 180),
        ),

    }

    VOWELS = "aeiou"

    CONSONANTS = {
        "l",
        "m",
        "n",
        "r",
        "y",
        "w",
        "b",
        "d",
        "g",
        "k",
        "p",
        "t",
        "s",
        "f",
        "h",
        "v",
        "z",
    }

    def synthesize(
        self,
        score,
        voicebank=None
    ):

        if not score.notes:

            return AudioBuffer(
                np.array([], dtype=np.float32),
                self.SAMPLE_RATE
            )

        rendered_notes = []

        previous_note = None

        for note in score.notes:

            audio = self.synthesize_note(
                note,
                previous_note
            )

            rendered_notes.append(audio)

            previous_note = note

        samples = np.concatenate(
            rendered_notes
        )

        samples = self.normalize(samples)

        return AudioBuffer(
            samples,
            self.SAMPLE_RATE
        )

    def synthesize_note(
            self,
            note,
            previous_note=None
    ):

        sample_count = max(
            1,
            int(note.duration * self.SAMPLE_RATE)
        )

        time = (
                np.arange(sample_count) /
                self.SAMPLE_RATE
        )

        vowel = self.extract_vowel(
            note.phoneme
        )

        consonant = self.extract_initial_consonant(
            note.phoneme
        )

        frequency = self.create_pitch_curve(
            note,
            previous_note,
            time
        )

        # Start with silence.
        signal = np.zeros(
            sample_count,
            dtype=np.float64
        )

        # -------------------------------------------------
        # 1. Consonant section
        # -------------------------------------------------

        consonant_duration = 0.0

        if consonant is not None:
            consonant_duration = min(
                0.14,
                note.duration * 0.30
            )

            consonant_samples = int(
                consonant_duration *
                self.SAMPLE_RATE
            )

            consonant_samples = min(
                consonant_samples,
                sample_count
            )

            consonant_signal = self.consonant_sound(
                consonant,
                consonant_samples
            )

            signal[
                :consonant_samples
            ] += consonant_signal

        # -------------------------------------------------
        # 2. Vowel section
        # -------------------------------------------------

        vowel_start = int(
            consonant_duration *
            self.SAMPLE_RATE
        )

        vowel_start = min(
            vowel_start,
            sample_count
        )

        vowel_time = time[
            vowel_start:
        ]

        vowel_frequency = frequency[
            vowel_start:
        ]

        if len(vowel_time) > 0:
            vowel_signal = self.source(
                vowel_time,
                vowel_frequency
            )

            vowel_signal = self.vocal_filter(
                vowel_signal,
                vowel
            )

            vowel_signal *= self.envelope(
                vowel_time,
                note.duration -
                consonant_duration
            )

            signal[
                vowel_start:
            ] += vowel_signal

        signal *= note.velocity

        return signal.astype(
            np.float32
        )

    def extract_vowel(
        self,
        phoneme
    ):

        phoneme = str(
            phoneme
        ).lower()

        for character in phoneme:

            if character in self.VOWELS:

                return character

        return "a"

    def extract_initial_consonant(
            self,
            phoneme
    ):

        phoneme = str(
            phoneme
        ).lower()

        for character in phoneme:

            if character in self.VOWELS:
                break

            if character in self.CONSONANTS:
                return character

        return None

    def create_pitch_curve(
        self,
        note,
        previous_note,
        time
    ):

        frequency = np.full(
            len(time),
            note.frequency,
            dtype=np.float64
        )

        if previous_note is not None:

            transition_time = min(
                0.06,
                note.duration / 5
            )

            if transition_time > 0:

                mask = time < transition_time

                progress = (
                    time[mask] /
                    transition_time
                )

                progress = (
                    progress *
                    progress *
                    (3 - 2 * progress)
                )

                frequency[mask] = (
                    previous_note.frequency +
                    (
                        note.frequency -
                        previous_note.frequency
                    ) *
                    progress
                )

        vibrato_start = min(
            0.18,
            note.duration / 3
        )

        vibrato_time = np.maximum(
            time - vibrato_start,
            0.0
        )

        vibrato_envelope = np.clip(
            vibrato_time / 0.25,
            0.0,
            1.0
        )

        vibrato = (
            1.0 +
            self.VIBRATO_DEPTH *
            vibrato_envelope *
            np.sin(
                2 *
                np.pi *
                self.VIBRATO_RATE *
                time
            )
        )

        frequency *= vibrato

        return frequency

    def source(
        self,
        time,
        frequency
    ):

        phase = np.cumsum(
            2 *
            np.pi *
            frequency /
            self.SAMPLE_RATE
        )

        signal = np.zeros_like(
            time,
            dtype=np.float64
        )

        for harmonic in range(
            1,
            self.HARMONICS + 1
        ):

            amplitude = (
                1.0 /
                harmonic ** 1.15
            )

            signal += (
                amplitude *
                np.sin(
                    harmonic *
                    phase
                )
            )

        peak = np.max(
            np.abs(signal)
        )

        if peak > 0:

            signal /= peak

        return signal

    def vocal_filter(
        self,
        signal,
        vowel
    ):

        formants = self.FORMANTS.get(
            vowel,
            self.FORMANTS["a"]
        )

        output = np.zeros_like(
            signal,
            dtype=np.float64
        )

        for frequency, bandwidth in formants:

            if frequency >= self.SAMPLE_RATE / 2:

                continue

            quality = (
                frequency /
                bandwidth
            )

            b, a = iirpeak(
                frequency,
                quality,
                fs=self.SAMPLE_RATE
            )

            filtered = lfilter(
                b,
                a,
                signal
            )

            output += filtered

        output *= 0.8

        original_peak = np.max(
            np.abs(signal)
        )

        output_peak = np.max(
            np.abs(output)
        )

        if output_peak > 0:

            output *= (
                original_peak /
                output_peak
            )

        return output

    def consonant_sound(
            self,
            consonant,
            sample_count
    ):

        if sample_count <= 0:
            return np.array(
                [],
                dtype=np.float64
            )

        rng = np.random.default_rng(1234)

        time = (
                np.arange(sample_count) /
                self.SAMPLE_RATE
        )

        decay = np.exp(
            -24.0 *
            time
        )

        noise = rng.normal(
            0.0,
            1.0,
            sample_count
        )

        if consonant in {
            "s",
            "f",
            "h"
        }:

            # Fricative: noise only.
            sound = noise * decay

        elif consonant in {
            "k",
            "t",
            "p"
        }:

            # Plosive: short strong burst.
            burst = np.exp(
                -70.0 *
                time
            )

            sound = noise * burst

        elif consonant in {
            "m",
            "n"
        }:

            # Nasal approximation.
            sound = (
                    0.7 *
                    np.sin(
                        2 *
                        np.pi *
                        180.0 *
                        time
                    ) *
                    decay
            )

        elif consonant in {
            "l",
            "r"
        }:

            # Liquid approximation.
            sound = (
                    0.45 *
                    np.sin(
                        2 *
                        np.pi *
                        220.0 *
                        time
                    ) *
                    decay
            )

        else:

            sound = noise * decay

        # Fade the consonant out smoothly.
        fade = np.linspace(
            1.0,
            0.0,
            sample_count
        )

        sound *= fade

        # Exaggerated for debugging.
        sound *= 0.8

        return sound.astype(
            np.float64
        )

    def envelope(
        self,
        time,
        duration
    ):

        attack = min(
            0.08,
            duration / 4
        )

        release = min(
            0.12,
            duration / 4
        )

        envelope = np.ones_like(
            time,
            dtype=np.float64
        )

        if attack > 0:

            attack_mask = time < attack

            envelope[attack_mask] = (
                time[attack_mask] /
                attack
            )

        if release > 0:

            release_mask = (
                time >
                duration - release
            )

            envelope[release_mask] = (
                (
                    duration -
                    time[release_mask]
                ) /
                release
            )

        return np.clip(
            envelope,
            0.0,
            1.0
        )

    def normalize(
        self,
        samples
    ):

        peak = np.max(
            np.abs(samples)
        )

        if peak == 0:

            return samples

        return (
            samples /
            peak *
            0.8
        )