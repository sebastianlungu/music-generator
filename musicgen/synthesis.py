"""
Audio synthesis for MusicGen using pure Python and FluidSynth.

This module handles:
- MIDI to audio rendering using pure Python synthesis (preferred)
- MIDI to audio rendering using SoundFonts and FluidSynth (fallback)
- Audio normalization and processing
- Format conversion (WAV to MP3)
"""

import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import mido

from .audio_types import (
    DEFAULT_SAMPLE_RATE,
    AudioCapability,
    AudioInfo,
    DependencyError,
    FluidSynthError,
    SynthesisError,
    detect_fluidsynth_path,
    is_capability_available,
    require_capability,
    warn_missing_capability,
)


# Enhanced FluidSynth detection with environment variable support
def _setup_fluidsynth_path():
    """Setup FluidSynth path from environment variables or detection."""
    fluidsynth_path = detect_fluidsynth_path()
    if fluidsynth_path and sys.platform == "win32":
        # Add FluidSynth path to system PATH for Windows
        lib_path = fluidsynth_path / "lib"
        bin_path = fluidsynth_path / "bin"

        current_path = os.environ.get("PATH", "")
        paths_to_add = []

        if lib_path.exists():
            paths_to_add.append(str(lib_path))
        if bin_path.exists():
            paths_to_add.append(str(bin_path))

        if paths_to_add:
            new_path = os.pathsep.join(paths_to_add + [current_path])
            os.environ["PATH"] = new_path


# Setup FluidSynth path before imports
_setup_fluidsynth_path()

# Handle pretty_midi import - fail fast when needed
try:
    import pretty_midi

    PRETTY_MIDI_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    PRETTY_MIDI_AVAILABLE = False
    # Store error for later fail-fast behavior
    _PRETTY_MIDI_ERROR = e

    # Create minimal stub for type hints only
    class _StubPrettyMIDI:
        class PrettyMIDI:
            pass
        class Instrument:
            pass
        class Note:
            pass

    pretty_midi = _StubPrettyMIDI()

# Handle FluidSynth import - fail fast when needed
try:
    import fluidsynth

    FLUIDSYNTH_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    FLUIDSYNTH_AVAILABLE = False
    # Store error for later fail-fast behavior
    _FLUIDSYNTH_ERROR = e

    # Create minimal stub for type hints only
    class _StubFluidSynth:
        class Synth:
            pass

    fluidsynth = _StubFluidSynth()

from .config import MusicGenConfig

# Set up logging
logger = logging.getLogger(__name__)


def check_fluidsynth_available() -> bool:
    """
    Check if FluidSynth is available for audio synthesis.

    Returns:
        True if FluidSynth is available, False otherwise
    """
    return is_capability_available(AudioCapability.AUDIO_SYNTHESIS)


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
    midi_data: "pretty_midi.PrettyMIDI",
    soundfont_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
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
        FluidSynthError: If FluidSynth-specific issues occur
    """
    if not check_fluidsynth_available():
        raise FluidSynthError(
            "FluidSynth is not available. "
            + "Install FluidSynth system package and pyfluidsynth. "
            + "Set FLUIDSYNTH_ROOT environment variable if needed."
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
                raise FluidSynthError(
                    f"Failed to load SoundFont: {soundfont_path}. "
                    f"Check if file is valid and accessible."
                )

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

    except FluidSynthError:
        raise  # Re-raise FluidSynth-specific errors
    except Exception as e:
        raise SynthesisError(f"Audio synthesis failed: {e}")


def _setup_midi_channels(
    synth: "fluidsynth.Synth", midi_data: pretty_midi.PrettyMIDI, soundfont_id: int
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
    synth: "fluidsynth.Synth", midi_data: pretty_midi.PrettyMIDI, sample_rate: int
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

    total_samples = (
        int(total_duration * sample_rate) + sample_rate
    )  # Add 1 second buffer

    # Collect all MIDI events with timing
    events = []

    for i, instrument in enumerate(midi_data.instruments):
        channel = 9 if instrument.is_drum else i % 16
        if channel == 9 and not instrument.is_drum:
            channel = (i + 1) % 16

        # Add note events
        for note in instrument.notes:
            # Note on
            events.append(
                {
                    "time": note.start,
                    "type": "note_on",
                    "channel": channel,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                }
            )

            # Note off
            events.append(
                {
                    "time": note.end,
                    "type": "note_off",
                    "channel": channel,
                    "pitch": note.pitch,
                    "velocity": 0,
                }
            )

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
    audio_data: np.ndarray, target_db: float = -1.0, fade_duration: float = 0.01
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
    audio_data: np.ndarray, fade_duration: float, sample_rate: int = 44100
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


def _midi_note_to_freq(note_number: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note_number - 69) / 12.0))


def _generate_sine_wave(
    frequency: float, duration: float, sample_rate: int = 44100, amplitude: float = 0.1
) -> np.ndarray:
    """Generate a sine wave for a given frequency and duration."""
    frames = int(duration * sample_rate)
    t = np.arange(frames) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * t)


def synthesize_midi_with_mido(
    midi_file_path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """
    Synthesize MIDI file to audio using mido and numpy only.

    This is a simple synthesizer that uses sine waves for each note.
    It works without requiring pretty_midi or FluidSynth dependencies.

    Args:
        midi_file_path: Path to MIDI file
        sample_rate: Audio sample rate

    Returns:
        Audio data as numpy array (mono)
    """
    logger.info("Using mido-based audio synthesis")

    # Load MIDI file
    mid = mido.MidiFile(str(midi_file_path))

    # Calculate total duration in seconds
    total_ticks = 0
    for track in mid.tracks:
        track_ticks = sum(msg.time for msg in track)
        total_ticks = max(total_ticks, track_ticks)

    # Convert ticks to seconds
    tempo = 500000  # Default tempo (120 BPM)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    seconds_per_tick = tempo / (1000000.0 * mid.ticks_per_beat)
    total_duration = total_ticks * seconds_per_tick

    if total_duration <= 0:
        total_duration = 4.0  # Default 4 seconds if no duration detected

    # Create audio buffer
    audio_frames = int(total_duration * sample_rate)
    audio_data = np.zeros(audio_frames)

    # Process each track
    for track in mid.tracks:
        # Track active notes and their start times
        active_notes = {}
        current_time = 0.0

        for msg in track:
            # Update current time
            current_time += msg.time * seconds_per_tick

            if msg.type == 'note_on' and msg.velocity > 0:
                # Start note
                active_notes[msg.note] = current_time

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # End note
                if msg.note in active_notes:
                    start_time = active_notes[msg.note]
                    duration = current_time - start_time

                    if duration > 0:
                        # Generate note audio
                        frequency = _midi_note_to_freq(msg.note)
                        note_audio = _generate_sine_wave(frequency, duration, sample_rate)

                        # Add to audio buffer
                        start_frame = int(start_time * sample_rate)
                        end_frame = start_frame + len(note_audio)

                        if end_frame <= len(audio_data):
                            audio_data[start_frame:end_frame] += note_audio

                    del active_notes[msg.note]

    # Normalize audio to prevent clipping
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8

    # Convert to stereo for consistency with other synthesis methods
    stereo_audio = np.column_stack([audio_data, audio_data])

    return stereo_audio.astype(np.float32)


def synthesize_midi(
    midi_data: "pretty_midi.PrettyMIDI", config: MusicGenConfig
) -> np.ndarray:
    """
    Synthesize MIDI data to audio according to configuration.

    Args:
        midi_data: MIDI data to synthesize
        config: Configuration containing synthesis parameters

    Returns:
        Synthesized audio data

    Raises:
        DependencyError: If audio synthesis dependencies are missing
        SynthesisError: If synthesis fails
        FluidSynthError: If FluidSynth-specific issues occur
    """
    # Require audio synthesis capability - fail fast if not available
    require_capability(AudioCapability.AUDIO_SYNTHESIS, "audio synthesis")

    # Try pure Python synthesis first (preferred method)
    try:
        import soundfile
        import scipy

        from .pure_synthesis import synthesize_pretty_midi

        logger.info("Using pure Python audio synthesis")
        raw_audio = synthesize_pretty_midi(midi_data, config.sample_rate)

    except (ImportError, Exception) as e:
        logger.info(f"Pure Python synthesis not available ({e}), trying FluidSynth")

        # Fallback to FluidSynth synthesis
        if config.soundfont_path is None:
            raise SynthesisError(
                "SoundFont required for FluidSynth synthesis. "
                "Provide a .sf2 file path in config.soundfont_path, "
                "or install pure Python synthesis dependencies: "
                "uv sync --extra audio-synthesis"
            )

        if not validate_soundfont(config.soundfont_path):
            raise FluidSynthError(f"Invalid SoundFont: {config.soundfont_path}")

        # Render MIDI to audio using FluidSynth
        raw_audio = render_midi_to_audio(
            midi_data, config.soundfont_path, config.sample_rate
        )

    # Normalize and process audio
    processed_audio = normalize_audio(
        raw_audio,
        target_db=-1.0,  # Leave some headroom
        fade_duration=0.01,  # 10ms fade
    )

    return processed_audio


def create_silent_audio(
    duration_seconds: float, sample_rate: int = 44100
) -> np.ndarray:
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


def get_audio_info(audio_data: np.ndarray, sample_rate: int) -> AudioInfo:
    """
    Get information about audio data.

    Args:
        audio_data: Audio data
        sample_rate: Sample rate

    Returns:
        AudioInfo object with detailed information
    """
    if audio_data.size == 0:
        return AudioInfo(duration=0.0, channels=0, samples=0, sample_rate=sample_rate)

    samples = len(audio_data)
    duration = samples / sample_rate
    channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1

    peak_level = float(np.abs(audio_data).max())
    rms_level = float(np.sqrt(np.mean(audio_data**2)))

    return AudioInfo(
        duration=duration,
        channels=channels,
        samples=samples,
        sample_rate=sample_rate,
        peak_level=peak_level,
        rms_level=rms_level,
    )
