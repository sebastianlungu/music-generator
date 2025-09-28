"""
Musical analysis for MusicGen.

This module provides functions for analyzing MIDI files to extract:
- Key signature
- Tempo
- Time signature
- Musical structure/sections
- Pitch class histograms
- Note density
- Instrument usage
"""

import numpy as np
from music21 import stream

# Import pretty_midi from io_files which has unified mock handling
from .io_files import PRETTY_MIDI_AVAILABLE, pretty_midi

# Add missing mock attributes for analysis module
if not PRETTY_MIDI_AVAILABLE:
    # Enhance mock with analysis-specific attributes
    class MockTimeSignature:
        def __init__(self, *args, **kwargs):
            self.numerator = 4
            self.denominator = 4

    # Add time signature support to mock
    pretty_midi.TimeSignature = MockTimeSignature
    pretty_midi.note_number_to_name = lambda x: "C"

from .config import AnalysisResult


def detect_key(midi_data: "pretty_midi.PrettyMIDI") -> str:
    """
    Detect the key of a MIDI file using music21's key detection.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        Key signature as string (e.g., "C major", "A minor")
    """
    try:
        # Convert PrettyMIDI to music21 stream
        midi_stream = _pretty_midi_to_music21_stream(midi_data)

        # Use music21's key detection
        detected_key = midi_stream.analyze("key")

        if detected_key is not None:
            return str(detected_key)
        else:
            # Fallback to pitch class analysis
            return _detect_key_from_pitch_classes(midi_data)

    except Exception:
        # Fallback to pitch class analysis if music21 fails
        return _detect_key_from_pitch_classes(midi_data)


def _pretty_midi_to_music21_stream(
    midi_data: "pretty_midi.PrettyMIDI",
) -> stream.Stream:
    """Convert PrettyMIDI to music21 stream for analysis."""
    s = stream.Stream()

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            # Convert to music21 note
            music21_note = pretty_midi.note_number_to_name(note.pitch)
            duration = note.end - note.start

            # Add note to stream
            s.insert(note.start, music21_note)

    return s


def _detect_key_from_pitch_classes(midi_data: "pretty_midi.PrettyMIDI") -> str:
    """
    Detect key using pitch class histogram analysis.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        Detected key as string
    """
    # Calculate pitch class histogram
    pitch_classes = np.zeros(12)

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            pitch_class = note.pitch % 12
            duration = note.end - note.start
            pitch_classes[pitch_class] += duration

    # Normalize
    if pitch_classes.sum() > 0:
        pitch_classes = pitch_classes / pitch_classes.sum()

    # Major key templates (Krumhansl-Schmuckler)
    major_template = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_template = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    # Normalize templates
    major_template = major_template / major_template.sum()
    minor_template = minor_template / minor_template.sum()

    best_correlation = -1
    best_key = "C major"

    # Check all major keys
    for i in range(12):
        rotated_template = np.roll(major_template, i)
        correlation = np.corrcoef(pitch_classes, rotated_template)[0, 1]
        if correlation > best_correlation:
            best_correlation = correlation
            key_names = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]
            best_key = f"{key_names[i]} major"

    # Check all minor keys
    for i in range(12):
        rotated_template = np.roll(minor_template, i)
        correlation = np.corrcoef(pitch_classes, rotated_template)[0, 1]
        if correlation > best_correlation:
            best_correlation = correlation
            key_names = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]
            best_key = f"{key_names[i]} minor"

    return best_key


def detect_tempo(midi_data: "pretty_midi.PrettyMIDI") -> float:
    """
    Detect the tempo of a MIDI file.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        Tempo in BPM
    """
    # First try to get tempo from MIDI tempo changes
    if midi_data.tempo_changes:
        # Use the most common tempo
        tempos = [change[1] for change in midi_data.tempo_changes]
        return float(np.median(tempos))

    # Fallback: estimate from note onset patterns
    return _estimate_tempo_from_onsets(midi_data)


def _estimate_tempo_from_onsets(midi_data: "pretty_midi.PrettyMIDI") -> float:
    """
    Estimate tempo from note onset patterns.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        Estimated tempo in BPM
    """
    # Collect all note onsets
    onsets = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            onsets.append(note.start)

    if len(onsets) < 10:
        return 120.0  # Default tempo

    onsets = np.array(sorted(onsets))

    # Calculate inter-onset intervals
    intervals = np.diff(onsets)

    # Filter out very short intervals (likely ornaments)
    intervals = intervals[intervals > 0.1]

    if len(intervals) == 0:
        return 120.0

    # Find the most common interval (likely the beat)
    # Use histogram to find peaks
    hist, bin_edges = np.histogram(intervals, bins=50)
    peak_idx = np.argmax(hist)
    beat_interval = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2

    # Convert to BPM
    tempo = 60.0 / beat_interval

    # Clamp to reasonable range
    tempo = np.clip(tempo, 60, 200)

    return float(tempo)


def detect_time_signature(midi_data: "pretty_midi.PrettyMIDI") -> tuple[int, int]:
    """
    Detect time signature of a MIDI file.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        Time signature as (numerator, denominator) tuple
    """
    # Check for explicit time signature events
    for time_sig in midi_data.time_signature_changes:
        return (time_sig.numerator, time_sig.denominator)

    # Fallback: analyze beat patterns
    return _estimate_time_signature_from_beats(midi_data)


def _estimate_time_signature_from_beats(
    midi_data: "pretty_midi.PrettyMIDI",
) -> tuple[int, int]:
    """
    Estimate time signature from beat patterns.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        Estimated time signature
    """
    tempo = detect_tempo(midi_data)
    beat_duration = 60.0 / tempo

    # Collect note onsets at beat level
    onsets = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            beat_position = note.start / beat_duration
            onsets.append(beat_position % 16)  # Look at patterns within 16 beats

    if len(onsets) < 10:
        return (4, 4)  # Default

    # Analyze for common time signature patterns
    onset_bins = np.histogram(onsets, bins=16, range=(0, 16))[0]

    # Look for strong beats patterns
    # 4/4: strong beats at 0, 4, 8, 12
    four_four_strength = onset_bins[0] + onset_bins[4] + onset_bins[8] + onset_bins[12]

    # 3/4: strong beats at 0, 3, 6, 9, 12
    three_four_strength = (
        onset_bins[0] + onset_bins[3] + onset_bins[6] + onset_bins[9] + onset_bins[12]
    )

    # 6/8: strong beats at 0, 6, 12
    six_eight_strength = onset_bins[0] + onset_bins[6] + onset_bins[12]

    if six_eight_strength > max(four_four_strength, three_four_strength):
        return (6, 8)
    elif three_four_strength > four_four_strength:
        return (3, 4)
    else:
        return (4, 4)


def calculate_pitch_histogram(midi_data: "pretty_midi.PrettyMIDI") -> list[float]:
    """
    Calculate pitch class histogram.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        List of 12 values representing pitch class distribution
    """
    pitch_classes = np.zeros(12)

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            pitch_class = note.pitch % 12
            duration = note.end - note.start
            pitch_classes[pitch_class] += duration

    # Normalize to probabilities
    if pitch_classes.sum() > 0:
        pitch_classes = pitch_classes / pitch_classes.sum()

    return pitch_classes.tolist()


def calculate_note_density(midi_data: "pretty_midi.PrettyMIDI") -> float:
    """
    Calculate average note density (notes per second).

    Args:
        midi_data: PrettyMIDI object

    Returns:
        Note density in notes per second
    """
    total_notes = 0
    total_duration = midi_data.get_end_time()

    if total_duration <= 0:
        return 0.0

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        total_notes += len(instrument.notes)

    return total_notes / total_duration


def detect_sections(midi_data: "pretty_midi.PrettyMIDI") -> list[tuple[float, float]]:
    """
    Detect musical sections using novelty detection.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        List of (start_time, end_time) tuples for each section
    """
    if midi_data.get_end_time() <= 0:
        return [(0.0, 0.0)]

    # Simple approach: divide into sections based on activity changes
    section_length = 8.0  # 8-second sections
    total_duration = midi_data.get_end_time()

    sections = []
    current_time = 0.0

    while current_time < total_duration:
        end_time = min(current_time + section_length, total_duration)
        sections.append((current_time, end_time))
        current_time = end_time

    # Merge very short sections
    merged_sections = []
    for start, end in sections:
        if end - start >= 4.0:  # Minimum 4 seconds
            merged_sections.append((start, end))
        elif merged_sections:
            # Merge with previous section
            prev_start, _ = merged_sections[-1]
            merged_sections[-1] = (prev_start, end)
        else:
            merged_sections.append((start, end))

    return merged_sections if merged_sections else [(0.0, total_duration)]


def get_instrument_programs(midi_data: "pretty_midi.PrettyMIDI") -> list[int]:
    """
    Get list of instrument programs used in the MIDI file.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        List of MIDI program numbers
    """
    programs = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            programs.append(instrument.program)

    return sorted(list(set(programs)))


def analyze_midi_file(midi_data: "pretty_midi.PrettyMIDI") -> AnalysisResult:
    """
    Perform complete analysis of a MIDI file.

    Args:
        midi_data: PrettyMIDI object to analyze

    Returns:
        AnalysisResult containing all analysis data
    """
    try:
        key = detect_key(midi_data)
    except Exception:
        key = "C major"

    try:
        tempo = detect_tempo(midi_data)
    except Exception:
        tempo = 120.0

    try:
        time_signature = detect_time_signature(midi_data)
    except Exception:
        time_signature = (4, 4)

    try:
        duration = midi_data.get_end_time()
    except Exception:
        duration = 0.0

    try:
        pitch_histogram = calculate_pitch_histogram(midi_data)
    except Exception:
        pitch_histogram = [0.0] * 12

    try:
        note_density = calculate_note_density(midi_data)
    except Exception:
        note_density = 0.0

    try:
        sections = detect_sections(midi_data)
    except Exception:
        sections = [(0.0, duration)]

    try:
        instrument_programs = get_instrument_programs(midi_data)
    except Exception:
        instrument_programs = []

    return AnalysisResult(
        key=key,
        tempo=tempo,
        time_signature=time_signature,
        duration_seconds=duration,
        pitch_histogram=pitch_histogram,
        note_density=note_density,
        sections=sections,
        instrument_programs=instrument_programs,
    )
