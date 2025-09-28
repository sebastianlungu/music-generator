"""
Musical arrangement generation for MusicGen.

This module provides rule-based algorithms for generating new musical
arrangements based on analysis of input MIDI files.
"""

import random

import numpy as np

from .config import AnalysisResult, ArrangementConfig, Instrument, MusicGenConfig

# Import pretty_midi from io_files which has unified mock handling
from .io_files import pretty_midi

# General MIDI program mappings
GM_PROGRAMS = {
    Instrument.PIANO: 0,
    Instrument.GUITAR: 24,
    Instrument.VIOLIN: 40,
    Instrument.CELLO: 42,
    Instrument.FLUTE: 73,
    Instrument.CLARINET: 71,
    Instrument.TRUMPET: 56,
    Instrument.SAXOPHONE: 64,
    Instrument.VOICE: 52,  # Choir
    Instrument.CHOIR: 52,
    Instrument.STRINGS: 48,  # String ensemble
    Instrument.BRASS: 61,  # Brass section
    Instrument.WOODWINDS: 68,  # Oboe as representative
    Instrument.ORGAN: 19,
    Instrument.BASS: 32,  # Acoustic bass
    Instrument.DRUMS: 0,  # Drum kit (channel 9)
}


class ScaleGenerator:
    """Generate scales and chords for different keys."""

    MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
    MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]

    @staticmethod
    def get_scale_notes(key: str, octave: int = 4) -> list[int]:
        """
        Get MIDI note numbers for a scale.

        Args:
            key: Key signature (e.g., "C major", "A minor")
            octave: Base octave

        Returns:
            List of MIDI note numbers
        """
        key_parts = key.split()
        root_name = key_parts[0]
        mode = key_parts[1] if len(key_parts) > 1 else "major"

        # Map note names to chromatic scale positions
        note_map = {
            "C": 0,
            "C#": 1,
            "Db": 1,
            "D": 2,
            "D#": 3,
            "Eb": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "Gb": 6,
            "G": 7,
            "G#": 8,
            "Ab": 8,
            "A": 9,
            "A#": 10,
            "Bb": 10,
            "B": 11,
        }

        root_pitch = note_map.get(root_name, 0)
        base_note = octave * 12 + root_pitch

        if mode.lower() == "major":
            intervals = ScaleGenerator.MAJOR_INTERVALS
        else:  # minor
            intervals = ScaleGenerator.MINOR_INTERVALS

        return [base_note + interval for interval in intervals]

    @staticmethod
    def get_chord(
        scale_notes: list[int], degree: int, chord_type: str = "triad"
    ) -> list[int]:
        """
        Generate a chord based on scale degree.

        Args:
            scale_notes: Scale notes
            degree: Scale degree (1-7)
            chord_type: Type of chord ("triad", "seventh")

        Returns:
            List of MIDI note numbers forming the chord
        """
        if not scale_notes or degree < 1 or degree > len(scale_notes):
            return []

        root_idx = (degree - 1) % len(scale_notes)
        chord = [scale_notes[root_idx]]

        # Add third
        third_idx = (root_idx + 2) % len(scale_notes)
        chord.append(scale_notes[third_idx])

        # Add fifth
        fifth_idx = (root_idx + 4) % len(scale_notes)
        chord.append(scale_notes[fifth_idx])

        if chord_type == "seventh":
            # Add seventh
            seventh_idx = (root_idx + 6) % len(scale_notes)
            chord.append(scale_notes[seventh_idx])

        return sorted(chord)


class RhythmGenerator:
    """Generate rhythmic patterns."""

    @staticmethod
    def generate_basic_rhythm(
        beats_per_measure: int, note_density: float, randomness: float = 0.1
    ) -> list[float]:
        """
        Generate basic rhythmic pattern.

        Args:
            beats_per_measure: Number of beats per measure
            note_density: Target notes per beat
            randomness: Amount of rhythmic variation

        Returns:
            List of note start times within one measure
        """
        beat_duration = 1.0  # Normalized to 1 beat = 1.0
        measure_duration = beats_per_measure * beat_duration

        # Calculate target number of notes
        target_notes = int(beats_per_measure * note_density)
        target_notes = max(
            1, min(target_notes, beats_per_measure * 4)
        )  # Reasonable limits

        note_times = []
        for i in range(target_notes):
            # Base position
            base_time = (i / target_notes) * measure_duration

            # Add rhythmic variation
            if randomness > 0:
                variation = random.uniform(-randomness, randomness) * beat_duration
                base_time += variation

            # Keep within measure bounds
            base_time = max(0, min(base_time, measure_duration - 0.1))
            note_times.append(base_time)

        return sorted(note_times)

    @staticmethod
    def quantize_to_grid(
        note_times: list[float], grid_size: float = 0.25
    ) -> list[float]:
        """
        Quantize note times to a rhythmic grid.

        Args:
            note_times: List of note start times
            grid_size: Grid size (0.25 = sixteenth note, 0.5 = eighth note)

        Returns:
            Quantized note times
        """
        return [round(time / grid_size) * grid_size for time in note_times]


class MelodyGenerator:
    """Generate melodic lines."""

    @staticmethod
    def generate_melody(
        scale_notes: list[int],
        rhythm_times: list[float],
        style: str = "stepwise",
        octave_range: int = 2,
    ) -> list[tuple[int, float]]:
        """
        Generate a melody line.

        Args:
            scale_notes: Available scale notes
            rhythm_times: Note timing
            style: Melodic style ("stepwise", "arpeggiated", "random")
            octave_range: Range of octaves to use

        Returns:
            List of (pitch, start_time) tuples
        """
        if not scale_notes or not rhythm_times:
            return []

        # Extend scale across octaves
        extended_scale = []
        for octave in range(octave_range):
            for note in scale_notes:
                extended_scale.append(note + octave * 12)

        melody = []
        current_note_idx = len(extended_scale) // 2  # Start in middle

        for time in rhythm_times:
            if style == "stepwise":
                # Move by steps
                direction = random.choice([-1, 0, 1])
                current_note_idx = max(
                    0, min(current_note_idx + direction, len(extended_scale) - 1)
                )
            elif style == "arpeggiated":
                # Jump by thirds or fourths
                jump = random.choice([-4, -2, 2, 4])
                current_note_idx = max(
                    0, min(current_note_idx + jump, len(extended_scale) - 1)
                )
            else:  # random
                current_note_idx = random.randint(0, len(extended_scale) - 1)

            pitch = extended_scale[current_note_idx]
            melody.append((pitch, time))

        return melody


class HarmonyGenerator:
    """Generate harmonic progressions and accompaniments."""

    COMMON_PROGRESSIONS = {
        "classical": [1, 4, 5, 1],  # I-IV-V-I
        "pop": [1, 5, 6, 4],  # I-V-vi-IV
        "jazz": [1, 6, 2, 5],  # I-vi-ii-V
        "modal": [1, 7, 4, 1],  # I-bVII-IV-I
    }

    @staticmethod
    def generate_chord_progression(
        scale_notes: list[int], measures: int, style: str = "classical"
    ) -> list[list[int]]:
        """
        Generate a chord progression.

        Args:
            scale_notes: Scale notes
            measures: Number of measures
            style: Harmonic style

        Returns:
            List of chord note lists
        """
        if not scale_notes:
            return []

        progression_template = HarmonyGenerator.COMMON_PROGRESSIONS.get(
            style, HarmonyGenerator.COMMON_PROGRESSIONS["classical"]
        )

        chords = []
        for measure in range(measures):
            degree = progression_template[measure % len(progression_template)]
            chord = ScaleGenerator.get_chord(scale_notes, degree)
            chords.append(chord)

        return chords

    @staticmethod
    def create_accompaniment_pattern(
        chords: list[list[int]], rhythm_times: list[float], pattern_type: str = "block"
    ) -> list[tuple[list[int], float]]:
        """
        Create accompaniment pattern from chords.

        Args:
            chords: List of chord note lists
            rhythm_times: Rhythm timing
            pattern_type: Pattern type ("block", "arpeggiated", "alberti")

        Returns:
            List of (chord_notes, start_time) tuples
        """
        if not chords or not rhythm_times:
            return []

        accompaniment = []
        measure_duration = 4.0  # Assume 4/4 time

        for i, time in enumerate(rhythm_times):
            # Determine which chord to use based on time
            measure = int(time // measure_duration)
            chord = chords[measure % len(chords)]

            if pattern_type == "block":
                # Play full chord
                accompaniment.append((chord, time))
            elif pattern_type == "arpeggiated":
                # Play chord notes in sequence
                note_idx = i % len(chord)
                accompaniment.append(([chord[note_idx]], time))
            elif pattern_type == "alberti":
                # Alberti bass pattern: low-high-middle-high
                if len(chord) >= 3:
                    pattern_idx = i % 4
                    if pattern_idx == 0:
                        note = chord[0]  # Low
                    elif pattern_idx == 1:
                        note = chord[2]  # High
                    elif pattern_idx == 2:
                        note = chord[1]  # Middle
                    else:
                        note = chord[2]  # High
                    accompaniment.append(([note], time))
                else:
                    accompaniment.append((chord, time))

        return accompaniment


def generate_arrangement(
    config: MusicGenConfig,
    analysis: AnalysisResult,
    arrangement_config: ArrangementConfig | None = None,
) -> "pretty_midi.PrettyMIDI":
    """
    Generate a complete musical arrangement.

    Args:
        config: Main configuration
        analysis: Analysis of input MIDI
        arrangement_config: Arrangement-specific configuration

    Returns:
        Generated PrettyMIDI object
    """
    if arrangement_config is None:
        arrangement_config = ArrangementConfig()

    # Set random seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Create new MIDI object
    midi_data = pretty_midi.PrettyMIDI()

    # Get effective tempo
    tempo = config.tempo_bpm if config.tempo_bpm else analysis.tempo
    if config.tempo_range:
        min_tempo, max_tempo = config.tempo_range
        tempo = random.uniform(min_tempo, max_tempo)

    # Get key
    key = config.key.value if config.key else analysis.key
    scale_notes = ScaleGenerator.get_scale_notes(key)

    # Calculate timing parameters
    beat_duration = 60.0 / tempo
    measures = int(config.duration_seconds / (beat_duration * 4)) + 1
    time_signature = analysis.time_signature

    # Generate rhythm patterns
    rhythm_times = RhythmGenerator.generate_basic_rhythm(
        time_signature[0], analysis.note_density, arrangement_config.humanization_amount
    )

    # Create instruments
    for i, instrument_type in enumerate(config.instruments[: config.voices]):
        instrument_program = GM_PROGRAMS.get(instrument_type, 0)

        # Create pretty_midi instrument
        if instrument_type == Instrument.DRUMS:
            instrument = pretty_midi.Instrument(program=0, is_drum=True)
            _add_drum_pattern(instrument, measures, beat_duration, config.style)
        else:
            instrument = pretty_midi.Instrument(program=instrument_program)

            if i == 0:  # Lead melody
                melody = MelodyGenerator.generate_melody(
                    scale_notes,
                    rhythm_times,
                    style="stepwise" if "classical" in config.style else "arpeggiated",
                )
                _add_melody_to_instrument(instrument, melody, beat_duration, measures)

            else:  # Harmony/accompaniment
                chords = HarmonyGenerator.generate_chord_progression(
                    scale_notes, measures, config.style
                )
                accompaniment = HarmonyGenerator.create_accompaniment_pattern(
                    chords,
                    rhythm_times,
                    "block" if "classical" in config.style else "arpeggiated",
                )
                _add_accompaniment_to_instrument(
                    instrument, accompaniment, beat_duration, measures
                )

        midi_data.instruments.append(instrument)

    # Ensure duration doesn't exceed limit
    _trim_to_duration(midi_data, config.duration_seconds)

    return midi_data


def _add_melody_to_instrument(
    instrument: pretty_midi.Instrument,
    melody: list[tuple[int, float]],
    beat_duration: float,
    measures: int,
) -> None:
    """Add melody notes to instrument."""
    measure_duration = beat_duration * 4

    for measure in range(measures):
        measure_start = measure * measure_duration

        for pitch, relative_time in melody:
            start_time = measure_start + relative_time * beat_duration
            end_time = start_time + beat_duration * 0.8  # Slight gap

            note = pretty_midi.Note(
                velocity=random.randint(70, 90),
                pitch=pitch,
                start=start_time,
                end=end_time,
            )
            instrument.notes.append(note)


def _add_accompaniment_to_instrument(
    instrument: pretty_midi.Instrument,
    accompaniment: list[tuple[list[int], float]],
    beat_duration: float,
    measures: int,
) -> None:
    """Add accompaniment notes to instrument."""
    measure_duration = beat_duration * 4

    for measure in range(measures):
        measure_start = measure * measure_duration

        for chord_notes, relative_time in accompaniment:
            start_time = measure_start + relative_time * beat_duration
            end_time = start_time + beat_duration * 1.5

            for pitch in chord_notes:
                note = pretty_midi.Note(
                    velocity=random.randint(50, 70),
                    pitch=pitch,
                    start=start_time,
                    end=end_time,
                )
                instrument.notes.append(note)


def _add_drum_pattern(
    instrument: pretty_midi.Instrument, measures: int, beat_duration: float, style: str
) -> None:
    """Add basic drum pattern."""
    # Basic kick and snare pattern
    kick_note = 36  # C2
    snare_note = 38  # D2
    hihat_note = 42  # F#2

    measure_duration = beat_duration * 4

    for measure in range(measures):
        measure_start = measure * measure_duration

        # Kick on beats 1 and 3
        for beat in [0, 2]:
            start_time = measure_start + beat * beat_duration
            note = pretty_midi.Note(
                velocity=80, pitch=kick_note, start=start_time, end=start_time + 0.1
            )
            instrument.notes.append(note)

        # Snare on beats 2 and 4
        for beat in [1, 3]:
            start_time = measure_start + beat * beat_duration
            note = pretty_midi.Note(
                velocity=70, pitch=snare_note, start=start_time, end=start_time + 0.1
            )
            instrument.notes.append(note)

        # Hi-hat on every beat
        for beat in range(4):
            start_time = measure_start + beat * beat_duration
            note = pretty_midi.Note(
                velocity=50, pitch=hihat_note, start=start_time, end=start_time + 0.05
            )
            instrument.notes.append(note)


def _trim_to_duration(midi_data: pretty_midi.PrettyMIDI, max_duration: float) -> None:
    """Trim MIDI data to maximum duration."""
    for instrument in midi_data.instruments:
        # Remove notes that start after max_duration
        instrument.notes = [
            note for note in instrument.notes if note.start < max_duration
        ]

        # Trim notes that extend beyond max_duration
        for note in instrument.notes:
            if note.end > max_duration:
                note.end = max_duration
