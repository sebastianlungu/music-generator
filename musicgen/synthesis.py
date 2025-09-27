"""
Audio synthesis for MusicGen using FluidSynth.

This module handles:
- MIDI to audio rendering using SoundFonts
- Audio normalization and processing
- Format conversion (WAV to MP3)
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import warnings

import numpy as np
import pretty_midi

try:
    import fluidsynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    FLUIDSYNTH_AVAILABLE = False
    warnings.warn(
        "FluidSynth not available. Audio synthesis will be disabled. "
        "Install with: pip install pyfluidsynth"
    )

from .config import MusicGenConfig


class SynthesisError(Exception):
    """Exception raised during audio synthesis."""
    pass


def check_fluidsynth_available() -> bool:
    """
    Check if FluidSynth is available for audio synthesis.

    Returns:
        True if FluidSynth is available, False otherwise
    """
    return FLUIDSYNTH_AVAILABLE


def validate_soundfont(soundfont_path: Path) -> bool:
    """
    Validate that a SoundFont file is accessible and readable.

    Args:
        soundfont_path: Path to SoundFont file

    Returns:
        True if valid, False otherwise
    """
    if not soundfont_path.exists():
        return False

    if not soundfont_path.is_file():
        return False

    if soundfont_path.suffix.lower() != ".sf2":
        return False

    try:
        # Try to read first few bytes to check if file is accessible
        with open(soundfont_path, "rb") as f:
            header = f.read(4)
            return header == b"RIFF"  # SoundFont files start with RIFF
    except Exception:
        return False


def render_midi_to_audio(
    midi_data: pretty_midi.PrettyMIDI,
    soundfont_path: Path,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Render MIDI data to audio using FluidSynth.

    Args:
        midi_data: PrettyMIDI object to render
        soundfont_path: Path to SoundFont file
        sample_rate: Audio sample rate

    Returns:
        Audio data as numpy array (shape: [samples] for mono, [samples, 2] for stereo)

    Raises:
        SynthesisError: If synthesis fails
    """
    if not FLUIDSYNTH_AVAILABLE:
        raise SynthesisError(
            "FluidSynth is not available. Install with: pip install pyfluidsynth"
        )

    if not validate_soundfont(soundfont_path):
        raise SynthesisError(f"Invalid SoundFont file: {soundfont_path}")

    try:
        # Create temporary MIDI file
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi:
            midi_data.write(temp_midi.name)
            temp_midi_path = temp_midi.name

        try:
            # Initialize FluidSynth
            fs = fluidsynth.Synth(samplerate=sample_rate)
            fs.start()

            # Load SoundFont
            sfid = fs.sfload(str(soundfont_path))
            if sfid == -1:
                raise SynthesisError(f"Failed to load SoundFont: {soundfont_path}")

            # Set up channels and instruments
            _setup_midi_channels(fs, midi_data, sfid)

            # Render MIDI to audio
            audio_data = _render_midi_events(fs, midi_data, sample_rate)

            # Clean up
            fs.delete()

            return audio_data

        finally:
            # Clean up temporary MIDI file
            try:
                os.unlink(temp_midi_path)
            except Exception:
                pass

    except Exception as e:
        raise SynthesisError(f"Audio synthesis failed: {e}")


def _setup_midi_channels(
    synth: "fluidsynth.Synth",
    midi_data: pretty_midi.PrettyMIDI,
    soundfont_id: int
) -> None:
    """
    Set up MIDI channels with appropriate instruments.

    Args:
        synth: FluidSynth synthesizer instance
        midi_data: MIDI data
        soundfont_id: SoundFont ID
    """
    for i, instrument in enumerate(midi_data.instruments):
        channel = 9 if instrument.is_drum else i % 16
        if channel == 9 and not instrument.is_drum:
            channel = (i + 1) % 16  # Skip drum channel for non-drum instruments

        # Set instrument program
        synth.program_select(channel, soundfont_id, instrument.program)


def _render_midi_events(
    synth: "fluidsynth.Synth",
    midi_data: pretty_midi.PrettyMIDI,
    sample_rate: int
) -> np.ndarray:
    """
    Render MIDI events to audio.

    Args:
        synth: FluidSynth synthesizer instance
        midi_data: MIDI data
        sample_rate: Audio sample rate

    Returns:
        Rendered audio data
    """
    # Calculate total duration and samples
    total_duration = midi_data.get_end_time()
    if total_duration <= 0:
        return np.zeros((1024, 2), dtype=np.float32)

    total_samples = int(total_duration * sample_rate) + sample_rate  # Add 1 second buffer

    # Collect all MIDI events with timing
    events = []

    for i, instrument in enumerate(midi_data.instruments):
        channel = 9 if instrument.is_drum else i % 16
        if channel == 9 and not instrument.is_drum:
            channel = (i + 1) % 16

        # Add note events
        for note in instrument.notes:
            # Note on
            events.append({
                "time": note.start,
                "type": "note_on",
                "channel": channel,
                "pitch": note.pitch,
                "velocity": note.velocity
            })

            # Note off
            events.append({
                "time": note.end,
                "type": "note_off",
                "channel": channel,
                "pitch": note.pitch,
                "velocity": 0
            })

    # Sort events by time
    events.sort(key=lambda x: x["time"])

    # Render audio in chunks
    chunk_duration = 0.1  # 100ms chunks
    chunk_samples = int(chunk_duration * sample_rate)
    audio_chunks = []

    current_time = 0.0
    event_idx = 0

    while current_time < total_duration:
        chunk_end_time = current_time + chunk_duration

        # Process events in this chunk
        while event_idx < len(events) and events[event_idx]["time"] < chunk_end_time:
            event = events[event_idx]

            if event["type"] == "note_on":
                synth.noteon(event["channel"], event["pitch"], event["velocity"])
            elif event["type"] == "note_off":
                synth.noteoff(event["channel"], event["pitch"])

            event_idx += 1

        # Render audio chunk
        chunk_audio = synth.get_samples(chunk_samples)
        audio_chunks.append(chunk_audio)

        current_time = chunk_end_time

    # Concatenate all chunks
    if audio_chunks:
        audio_data = np.concatenate(audio_chunks, axis=0)
    else:
        audio_data = np.zeros((total_samples, 2), dtype=np.float32)

    return audio_data


def normalize_audio(
    audio_data: np.ndarray,
    target_db: float = -1.0,
    fade_duration: float = 0.01
) -> np.ndarray:
    """
    Normalize audio to target level and apply fade in/out.

    Args:
        audio_data: Input audio data
        target_db: Target level in dB
        fade_duration: Fade in/out duration in seconds

    Returns:
        Normalized audio data
    """
    if audio_data.size == 0:
        return audio_data

    # Convert to float32 if needed
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Find peak level
    peak = np.abs(audio_data).max()
    if peak == 0:
        return audio_data

    # Calculate normalization factor
    target_linear = 10 ** (target_db / 20.0)
    normalization_factor = target_linear / peak

    # Apply normalization
    normalized_audio = audio_data * normalization_factor

    # Apply fade in/out
    if fade_duration > 0 and len(normalized_audio) > 0:
        normalized_audio = _apply_fade(normalized_audio, fade_duration)

    return normalized_audio


def _apply_fade(
    audio_data: np.ndarray,
    fade_duration: float,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply fade in and fade out to audio.

    Args:
        audio_data: Input audio data
        fade_duration: Fade duration in seconds
        sample_rate: Audio sample rate

    Returns:
        Audio with fades applied
    """
    fade_samples = int(fade_duration * sample_rate)
    audio_length = len(audio_data)

    if fade_samples >= audio_length // 2:
        fade_samples = audio_length // 4

    if fade_samples <= 0:
        return audio_data

    # Create fade curves
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    # Apply fade in
    if len(audio_data.shape) == 1:  # Mono
        audio_data[:fade_samples] *= fade_in
        audio_data[-fade_samples:] *= fade_out
    else:  # Stereo
        audio_data[:fade_samples, :] *= fade_in[:, np.newaxis]
        audio_data[-fade_samples:, :] *= fade_out[:, np.newaxis]

    return audio_data


def synthesize_midi(
    midi_data: pretty_midi.PrettyMIDI,
    config: MusicGenConfig
) -> Optional[np.ndarray]:
    """
    Synthesize MIDI data to audio according to configuration.

    Args:
        midi_data: MIDI data to synthesize
        config: Configuration containing synthesis parameters

    Returns:
        Synthesized audio data, or None if synthesis is not possible

    Raises:
        SynthesisError: If synthesis fails
    """
    if not FLUIDSYNTH_AVAILABLE:
        warnings.warn("FluidSynth not available, skipping audio synthesis")
        return None

    if config.soundfont_path is None:
        warnings.warn("No SoundFont specified, skipping audio synthesis")
        return None

    if not validate_soundfont(config.soundfont_path):
        raise SynthesisError(f"Invalid SoundFont: {config.soundfont_path}")

    # Render MIDI to audio
    raw_audio = render_midi_to_audio(
        midi_data,
        config.soundfont_path,
        config.sample_rate
    )

    # Normalize and process audio
    processed_audio = normalize_audio(
        raw_audio,
        target_db=-1.0,  # Leave some headroom
        fade_duration=0.01  # 10ms fade
    )

    return processed_audio


def create_silent_audio(duration_seconds: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Create silent audio for testing or fallback purposes.

    Args:
        duration_seconds: Duration in seconds
        sample_rate: Audio sample rate

    Returns:
        Silent audio data (stereo)
    """
    samples = int(duration_seconds * sample_rate)
    return np.zeros((samples, 2), dtype=np.float32)


def get_audio_info(audio_data: np.ndarray, sample_rate: int) -> dict:
    """
    Get information about audio data.

    Args:
        audio_data: Audio data
        sample_rate: Sample rate

    Returns:
        Dictionary with audio information
    """
    if audio_data.size == 0:
        return {
            "duration": 0.0,
            "channels": 0,
            "samples": 0,
            "peak_level": 0.0,
            "rms_level": 0.0
        }

    samples = len(audio_data)
    duration = samples / sample_rate
    channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1

    peak_level = float(np.abs(audio_data).max())
    rms_level = float(np.sqrt(np.mean(audio_data ** 2)))

    return {
        "duration": duration,
        "channels": channels,
        "samples": samples,
        "peak_level": peak_level,
        "rms_level": rms_level,
        "peak_db": 20 * np.log10(peak_level) if peak_level > 0 else -np.inf,
        "rms_db": 20 * np.log10(rms_level) if rms_level > 0 else -np.inf
    }