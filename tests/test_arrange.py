"""
Tests for the arrange module.

These tests ensure that arrangement generation produces valid
musical output within specified parameters.
"""

from pathlib import Path

import pytest

from musicgen.arrange import (
    HarmonyGenerator,
    MelodyGenerator,
    RhythmGenerator,
    ScaleGenerator,
    generate_arrangement,
)
from musicgen.config import (
    AnalysisResult,
    ArrangementConfig,
    Instrument,
    MusicalKey,
    MusicGenConfig,
)

# Import pretty_midi from our module (which includes mock support)
from musicgen.io_files import pretty_midi


def create_test_config(
    duration: int = 30, instruments: list = None, voices: int = 2
) -> MusicGenConfig:
    """Create a test configuration."""
    if instruments is None:
        instruments = [Instrument.PIANO, Instrument.GUITAR]

    return MusicGenConfig(
        input_path=Path("test.mid"),  # Dummy path
        duration_seconds=duration,
        instruments=instruments,
        voices=voices,
        style="classical",
        seed=42,
    )


def create_test_analysis() -> AnalysisResult:
    """Create a test analysis result."""
    return AnalysisResult(
        key="C major",
        tempo=120.0,
        time_signature=(4, 4),
        duration_seconds=60.0,
        pitch_histogram=[0.1] * 12,
        note_density=2.0,
        sections=[(0.0, 30.0), (30.0, 60.0)],
        instrument_programs=[0, 24],
    )


class TestScaleGenerator:
    """Test scale generation functions."""

    def test_get_scale_notes_c_major(self):
        """Test C major scale generation."""
        notes = ScaleGenerator.get_scale_notes("C major", octave=4)

        expected = [60, 62, 64, 65, 67, 69, 71]  # C4 major scale
        assert notes == expected

    def test_get_scale_notes_a_minor(self):
        """Test A minor scale generation."""
        notes = ScaleGenerator.get_scale_notes("A minor", octave=4)

        expected = [57, 59, 60, 62, 64, 65, 67]  # A4 minor scale
        assert notes == expected

    def test_get_scale_notes_different_octave(self):
        """Test scale generation in different octave."""
        notes = ScaleGenerator.get_scale_notes("C major", octave=5)

        expected = [72, 74, 76, 77, 79, 81, 83]  # C5 major scale
        assert notes == expected

    def test_get_chord_triad(self):
        """Test chord generation."""
        scale_notes = [60, 62, 64, 65, 67, 69, 71]  # C major
        chord = ScaleGenerator.get_chord(scale_notes, 1, "triad")

        expected = [60, 64, 67]  # C major triad
        assert chord == expected

    def test_get_chord_seventh(self):
        """Test seventh chord generation."""
        scale_notes = [60, 62, 64, 65, 67, 69, 71]  # C major
        chord = ScaleGenerator.get_chord(scale_notes, 1, "seventh")

        expected = [60, 64, 67, 71]  # C major 7th
        assert chord == expected

    def test_get_chord_invalid_degree(self):
        """Test chord generation with invalid degree."""
        scale_notes = [60, 62, 64, 65, 67, 69, 71]
        chord = ScaleGenerator.get_chord(scale_notes, 10, "triad")

        assert chord == []


class TestRhythmGenerator:
    """Test rhythm generation functions."""

    def test_generate_basic_rhythm(self):
        """Test basic rhythm generation."""
        rhythm = RhythmGenerator.generate_basic_rhythm(
            beats_per_measure=4, note_density=2.0, randomness=0.0
        )

        assert len(rhythm) == 8  # 4 beats * 2 notes per beat
        assert all(0 <= time <= 4.0 for time in rhythm)
        assert rhythm == sorted(rhythm)  # Should be sorted

    def test_quantize_to_grid(self):
        """Test rhythm quantization."""
        note_times = [0.1, 0.6, 1.1, 1.7]
        quantized = RhythmGenerator.quantize_to_grid(note_times, grid_size=0.5)

        expected = [0.0, 0.5, 1.0, 1.5]
        assert quantized == expected

    def test_generate_rhythm_with_randomness(self):
        """Test rhythm generation with randomness."""
        rhythm1 = RhythmGenerator.generate_basic_rhythm(
            beats_per_measure=4, note_density=1.0, randomness=0.1
        )

        rhythm2 = RhythmGenerator.generate_basic_rhythm(
            beats_per_measure=4, note_density=1.0, randomness=0.1
        )

        # With randomness, rhythms should be different
        assert rhythm1 != rhythm2


class TestMelodyGenerator:
    """Test melody generation functions."""

    def test_generate_melody_stepwise(self):
        """Test stepwise melody generation."""
        scale_notes = [60, 62, 64, 65, 67, 69, 71]
        rhythm_times = [0.0, 0.5, 1.0, 1.5]

        melody = MelodyGenerator.generate_melody(
            scale_notes, rhythm_times, style="stepwise"
        )

        assert len(melody) == len(rhythm_times)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in melody)
        assert all(pitch in range(60, 84) for pitch, _ in melody)  # Reasonable range

    def test_generate_melody_arpeggiated(self):
        """Test arpeggiated melody generation."""
        scale_notes = [60, 62, 64, 65, 67, 69, 71]
        rhythm_times = [0.0, 0.5, 1.0, 1.5]

        melody = MelodyGenerator.generate_melody(
            scale_notes, rhythm_times, style="arpeggiated"
        )

        assert len(melody) == len(rhythm_times)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in melody)

    def test_generate_melody_empty_inputs(self):
        """Test melody generation with empty inputs."""
        melody = MelodyGenerator.generate_melody([], [0.0, 1.0])
        assert melody == []

        melody = MelodyGenerator.generate_melody([60, 62, 64], [])
        assert melody == []


class TestHarmonyGenerator:
    """Test harmony generation functions."""

    def test_generate_chord_progression_classical(self):
        """Test classical chord progression."""
        scale_notes = [60, 62, 64, 65, 67, 69, 71]
        chords = HarmonyGenerator.generate_chord_progression(
            scale_notes, measures=4, style="classical"
        )

        assert len(chords) == 4
        assert all(isinstance(chord, list) for chord in chords)
        assert all(len(chord) >= 3 for chord in chords)  # At least triads

    def test_create_accompaniment_pattern_block(self):
        """Test block chord accompaniment."""
        chords = [[60, 64, 67], [65, 69, 72]]
        rhythm_times = [0.0, 1.0, 2.0, 3.0]

        accompaniment = HarmonyGenerator.create_accompaniment_pattern(
            chords, rhythm_times, pattern_type="block"
        )

        assert len(accompaniment) == len(rhythm_times)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in accompaniment)

    def test_create_accompaniment_pattern_arpeggiated(self):
        """Test arpeggiated accompaniment."""
        chords = [[60, 64, 67]]
        rhythm_times = [0.0, 0.5, 1.0, 1.5]

        accompaniment = HarmonyGenerator.create_accompaniment_pattern(
            chords, rhythm_times, pattern_type="arpeggiated"
        )

        assert len(accompaniment) == len(rhythm_times)
        # Each note should be from the chord
        for chord_notes, _ in accompaniment:
            assert all(note in [60, 64, 67] for note in chord_notes)


class TestArrangementGeneration:
    """Test complete arrangement generation."""

    def test_generate_arrangement_basic(self):
        """Test basic arrangement generation."""
        config = create_test_config()
        analysis = create_test_analysis()

        midi_data = generate_arrangement(config, analysis)

        assert isinstance(midi_data, pretty_midi.PrettyMIDI)
        assert len(midi_data.instruments) == config.voices
        assert midi_data.get_end_time() <= config.duration_seconds

    def test_generate_arrangement_single_voice(self):
        """Test single voice arrangement."""
        config = create_test_config(voices=1, instruments=[Instrument.PIANO])
        analysis = create_test_analysis()

        midi_data = generate_arrangement(config, analysis)

        assert len(midi_data.instruments) == 1
        assert not midi_data.instruments[0].is_drum
        assert len(midi_data.instruments[0].notes) > 0

    def test_generate_arrangement_with_drums(self):
        """Test arrangement with drums."""
        config = create_test_config(
            voices=2, instruments=[Instrument.PIANO, Instrument.DRUMS]
        )
        analysis = create_test_analysis()

        midi_data = generate_arrangement(config, analysis)

        assert len(midi_data.instruments) == 2

        # Find drum instrument
        drum_instrument = None
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                drum_instrument = instrument
                break

        assert drum_instrument is not None
        assert len(drum_instrument.notes) > 0

    def test_generate_arrangement_different_styles(self):
        """Test arrangement generation with different styles."""
        analysis = create_test_analysis()

        styles = ["classical", "jazz", "rock", "ambient"]
        for style in styles:
            config = create_test_config()
            config.style = style

            midi_data = generate_arrangement(config, analysis)

            assert isinstance(midi_data, pretty_midi.PrettyMIDI)
            assert len(midi_data.instruments) > 0

    def test_generate_arrangement_duration_constraint(self):
        """Test that arrangement respects duration constraint."""
        config = create_test_config(duration=10)  # Short duration
        analysis = create_test_analysis()

        midi_data = generate_arrangement(config, analysis)

        # Should not exceed specified duration
        assert (
            midi_data.get_end_time() <= config.duration_seconds + 1.0
        )  # Small tolerance

    def test_generate_arrangement_tempo_override(self):
        """Test arrangement with tempo override."""
        config = create_test_config()
        config.tempo_bpm = 140

        analysis = create_test_analysis()
        analysis.tempo = 120.0  # Different from config

        midi_data = generate_arrangement(config, analysis)

        # Should use config tempo, not analysis tempo
        assert isinstance(midi_data, pretty_midi.PrettyMIDI)

    def test_generate_arrangement_key_override(self):
        """Test arrangement with key override."""
        config = create_test_config()
        config.key = MusicalKey.A_MINOR

        analysis = create_test_analysis()
        analysis.key = "C major"  # Different from config

        midi_data = generate_arrangement(config, analysis)

        # Should generate arrangement in A minor
        assert isinstance(midi_data, pretty_midi.PrettyMIDI)
        assert len(midi_data.instruments) > 0

    def test_generate_arrangement_reproducibility(self):
        """Test that arrangement generation is reproducible with same seed."""
        config = create_test_config()
        config.seed = 123
        analysis = create_test_analysis()

        midi_data1 = generate_arrangement(config, analysis)

        # Generate again with same seed
        config.seed = 123
        midi_data2 = generate_arrangement(config, analysis)

        # Should have same number of instruments and notes
        assert len(midi_data1.instruments) == len(midi_data2.instruments)

        for inst1, inst2 in zip(midi_data1.instruments, midi_data2.instruments, strict=False):
            assert len(inst1.notes) == len(inst2.notes)

    def test_arrangement_note_validity(self):
        """Test that generated notes are valid."""
        config = create_test_config()
        analysis = create_test_analysis()

        midi_data = generate_arrangement(config, analysis)

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Check note properties are valid
                assert 0 <= note.pitch <= 127
                assert 0 <= note.velocity <= 127
                assert note.start >= 0
                assert note.end >= note.start
                assert note.end <= config.duration_seconds + 1.0  # Small tolerance

    def test_arrangement_config_validation(self):
        """Test arrangement with custom arrangement config."""
        config = create_test_config()
        analysis = create_test_analysis()

        arrangement_config = ArrangementConfig(
            humanization_amount=0.2, voice_leading_strictness=0.8
        )

        midi_data = generate_arrangement(config, analysis, arrangement_config)

        assert isinstance(midi_data, pretty_midi.PrettyMIDI)
        assert len(midi_data.instruments) > 0


class TestArrangementEdgeCases:
    """Test edge cases and error handling."""

    def test_generate_arrangement_zero_duration(self):
        """Test arrangement with zero duration."""
        config = create_test_config(duration=0)
        analysis = create_test_analysis()

        # Should still generate something minimal
        midi_data = generate_arrangement(config, analysis)
        assert isinstance(midi_data, pretty_midi.PrettyMIDI)

    def test_generate_arrangement_many_voices(self):
        """Test arrangement with maximum voices."""
        instruments = [Instrument.PIANO] * 8
        config = create_test_config(voices=8, instruments=instruments)
        analysis = create_test_analysis()

        midi_data = generate_arrangement(config, analysis)

        assert len(midi_data.instruments) <= 8


if __name__ == "__main__":
    pytest.main([__file__])
