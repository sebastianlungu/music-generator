"""
Tests for the analysis module.

These tests ensure that MIDI analysis functions correctly detect
key, tempo, time signature, and other musical features.
"""

import pytest

from musicgen.analysis import (
    analyze_midi_file,
    calculate_note_density,
    calculate_pitch_histogram,
    detect_key,
    detect_sections,
    detect_tempo,
    detect_time_signature,
    get_instrument_programs,
)
from musicgen.config import AnalysisResult

# Import pretty_midi from our module (which includes mock support)
from musicgen.io_files import pretty_midi


def create_test_midi(
    key: str = "C major",
    tempo: float = 120.0,
    time_signature: tuple = (4, 4),
    duration: float = 4.0,
    notes: list = None,
) -> pretty_midi.PrettyMIDI:
    """
    Create a test MIDI file with specified parameters.

    Args:
        key: Musical key
        tempo: Tempo in BPM
        time_signature: Time signature as (numerator, denominator)
        duration: Duration in seconds
        notes: List of (pitch, start, end) tuples

    Returns:
        PrettyMIDI object
    """
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Add time signature
    midi_data.time_signature_changes.append(
        pretty_midi.TimeSignature(time_signature[0], time_signature[1], 0)
    )

    # Create instrument
    instrument = pretty_midi.Instrument(program=0)  # Piano

    # Add notes
    if notes is None:
        # Create simple C major scale
        scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
        notes = []
        for i, pitch in enumerate(scale_notes):
            start_time = i * 0.5
            end_time = start_time + 0.4
            if end_time <= duration:
                notes.append((pitch, start_time, end_time))

    for pitch, start, end in notes:
        note = pretty_midi.Note(
            velocity=80, pitch=pitch, start=start, end=min(end, duration)
        )
        instrument.notes.append(note)

    midi_data.instruments.append(instrument)
    return midi_data


class TestAnalysis:
    """Test analysis functions."""

    def test_analyze_midi_file_basic(self):
        """Test basic MIDI file analysis."""
        midi_data = create_test_midi()
        result = analyze_midi_file(midi_data)

        assert isinstance(result, AnalysisResult)
        assert result.key is not None
        assert result.tempo > 0
        assert len(result.time_signature) == 2
        assert result.duration_seconds > 0
        assert len(result.pitch_histogram) == 12
        assert result.note_density >= 0
        assert len(result.sections) > 0

    def test_detect_key_c_major(self):
        """Test key detection for C major."""
        # Create MIDI with strong C major characteristics
        notes = [
            (60, 0.0, 1.0),  # C
            (64, 1.0, 2.0),  # E
            (67, 2.0, 3.0),  # G
            (72, 3.0, 4.0),  # C
        ]
        midi_data = create_test_midi(notes=notes)
        key = detect_key(midi_data)

        assert "major" in key.lower()

    def test_detect_tempo(self):
        """Test tempo detection."""
        expected_tempo = 140.0
        midi_data = create_test_midi(tempo=expected_tempo)
        detected_tempo = detect_tempo(midi_data)

        # Should be close to expected tempo
        assert abs(detected_tempo - expected_tempo) < 10

    def test_detect_time_signature(self):
        """Test time signature detection."""
        expected_sig = (3, 4)
        midi_data = create_test_midi(time_signature=expected_sig)
        detected_sig = detect_time_signature(midi_data)

        assert detected_sig == expected_sig

    def test_calculate_pitch_histogram(self):
        """Test pitch class histogram calculation."""
        # Create MIDI with only C notes
        notes = [(60, 0.0, 1.0), (72, 1.0, 2.0)]  # C4 and C5
        midi_data = create_test_midi(notes=notes)
        histogram = calculate_pitch_histogram(midi_data)

        assert len(histogram) == 12
        assert histogram[0] > 0  # C should have non-zero weight
        assert sum(histogram) == pytest.approx(1.0, abs=1e-6)  # Should sum to 1

    def test_calculate_note_density(self):
        """Test note density calculation."""
        # Create MIDI with known note density
        notes = [
            (60, 0.0, 0.5),
            (62, 0.5, 1.0),
            (64, 1.0, 1.5),
            (65, 1.5, 2.0),
        ]
        midi_data = create_test_midi(notes=notes, duration=2.0)
        density = calculate_note_density(midi_data)

        expected_density = 4 / 2.0  # 4 notes in 2 seconds
        assert density == expected_density

    def test_detect_sections(self):
        """Test section detection."""
        midi_data = create_test_midi(duration=16.0)
        sections = detect_sections(midi_data)

        assert len(sections) > 0
        assert all(
            isinstance(section, tuple) and len(section) == 2 for section in sections
        )
        assert all(start < end for start, end in sections)

    def test_get_instrument_programs(self):
        """Test instrument program detection."""
        midi_data = create_test_midi()

        # Add another instrument
        guitar = pretty_midi.Instrument(program=24)  # Guitar
        note = pretty_midi.Note(velocity=80, pitch=60, start=0, end=1)
        guitar.notes.append(note)
        midi_data.instruments.append(guitar)

        programs = get_instrument_programs(midi_data)

        assert 0 in programs  # Piano
        assert 24 in programs  # Guitar
        assert len(programs) == 2

    def test_empty_midi_file(self):
        """Test analysis of empty MIDI file."""
        midi_data = pretty_midi.PrettyMIDI()
        result = analyze_midi_file(midi_data)

        assert isinstance(result, AnalysisResult)
        assert result.duration_seconds == 0.0
        assert result.note_density == 0.0
        assert all(h == 0.0 for h in result.pitch_histogram)

    def test_drum_track_ignored(self):
        """Test that drum tracks are ignored in analysis."""
        midi_data = create_test_midi()

        # Add drum track
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        drum_note = pretty_midi.Note(velocity=80, pitch=36, start=0, end=1)
        drums.notes.append(drum_note)
        midi_data.instruments.append(drums)

        programs = get_instrument_programs(midi_data)
        assert len(programs) == 1  # Only piano, drums ignored

        # Pitch histogram should not include drum notes
        histogram = calculate_pitch_histogram(midi_data)
        # Should still be based only on piano notes
        assert len(histogram) == 12  # Should still have 12 pitch classes


class TestAnalysisEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_midi(self):
        """Test analysis of very short MIDI file."""
        notes = [(60, 0.0, 0.1)]
        midi_data = create_test_midi(notes=notes, duration=0.1)
        result = analyze_midi_file(midi_data)

        assert result.duration_seconds == pytest.approx(0.1, abs=0.01)
        assert result.note_density > 0

    def test_single_note_midi(self):
        """Test analysis of MIDI with single note."""
        notes = [(60, 0.0, 1.0)]
        midi_data = create_test_midi(notes=notes)
        result = analyze_midi_file(midi_data)

        assert result.key is not None
        assert result.tempo > 0
        assert result.note_density == 1.0  # 1 note per second

    def test_no_tempo_changes(self):
        """Test tempo detection when no tempo changes exist."""
        midi_data = pretty_midi.PrettyMIDI()  # No initial tempo set
        instrument = pretty_midi.Instrument(program=0)

        # Add notes with regular spacing
        for i in range(4):
            note = pretty_midi.Note(
                velocity=80, pitch=60, start=i * 0.5, end=i * 0.5 + 0.4
            )
            instrument.notes.append(note)

        midi_data.instruments.append(instrument)
        tempo = detect_tempo(midi_data)

        assert 60 <= tempo <= 200  # Should be in reasonable range

    def test_analysis_robustness(self):
        """Test that analysis doesn't crash on malformed data."""
        midi_data = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        # Add notes with unusual properties
        unusual_notes = [
            (0, 0.0, 1.0),  # Very low pitch
            (127, 1.0, 2.0),  # Very high pitch
            (60, 2.0, 2.0),  # Zero duration
            (60, 3.0, 2.5),  # Negative duration (end < start)
        ]

        for pitch, start, end in unusual_notes:
            note = pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
            instrument.notes.append(note)

        midi_data.instruments.append(instrument)

        # Should not crash
        result = analyze_midi_file(midi_data)
        assert isinstance(result, AnalysisResult)


if __name__ == "__main__":
    pytest.main([__file__])
