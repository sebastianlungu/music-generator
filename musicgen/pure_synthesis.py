"""
Pure Python Audio Synthesis for MusicGen.

This module provides MIDI to audio synthesis using only Python and numpy,
eliminating the need for external dependencies like FluidSynth or SoundFont files.

Features:
- Synthetic waveform generation (sine, sawtooth, square, triangle)
- ADSR envelope processing
- Polyphonic voice management
- GM instrument mapping to synthesis parameters
- High-quality audio output
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .audio_types import DEFAULT_SAMPLE_RATE


class WaveformType(Enum):
    """Supported waveform types for synthesis."""

    SINE = "sine"
    SAWTOOTH = "sawtooth"
    SQUARE = "square"
    TRIANGLE = "triangle"
    NOISE = "noise"


@dataclass
class ADSREnvelope:
    """ADSR envelope parameters."""

    attack: float = 0.01  # Attack time in seconds
    decay: float = 0.1  # Decay time in seconds
    sustain: float = 0.7  # Sustain level (0-1)
    release: float = 0.3  # Release time in seconds


@dataclass
class InstrumentProfile:
    """Synthesis parameters for an instrument."""

    name: str
    waveform: WaveformType
    envelope: ADSREnvelope
    harmonics: list[tuple[float, float]] = None  # (frequency_ratio, amplitude)
    filter_cutoff: float = 1.0  # Relative to fundamental frequency
    vibrato_rate: float = 0.0  # Hz
    vibrato_depth: float = 0.0  # Semitones


# General MIDI instrument profiles
GM_INSTRUMENTS = {
    # Piano family (0-7)
    0: InstrumentProfile(
        "Acoustic Grand Piano",
        WaveformType.SINE,
        ADSREnvelope(0.01, 0.3, 0.4, 0.8),
        [(1.0, 1.0), (2.0, 0.3), (3.0, 0.1)],
    ),
    1: InstrumentProfile(
        "Bright Acoustic Piano",
        WaveformType.SINE,
        ADSREnvelope(0.005, 0.2, 0.5, 0.6),
        [(1.0, 1.0), (2.0, 0.4), (4.0, 0.2)],
    ),
    2: InstrumentProfile(
        "Electric Grand Piano",
        WaveformType.SINE,
        ADSREnvelope(0.01, 0.4, 0.6, 0.7),
        [(1.0, 1.0), (2.0, 0.2)],
    ),
    # Organ family (16-23)
    16: InstrumentProfile(
        "Drawbar Organ",
        WaveformType.SINE,
        ADSREnvelope(0.01, 0.0, 1.0, 0.1),
        [(1.0, 1.0), (2.0, 0.8), (3.0, 0.6), (4.0, 0.4)],
    ),
    17: InstrumentProfile(
        "Percussive Organ",
        WaveformType.SINE,
        ADSREnvelope(0.001, 0.1, 0.3, 0.2),
        [(1.0, 1.0), (2.0, 0.5)],
    ),
    18: InstrumentProfile(
        "Rock Organ", WaveformType.SAWTOOTH, ADSREnvelope(0.01, 0.0, 1.0, 0.1)
    ),
    # Guitar family (24-31)
    24: InstrumentProfile(
        "Acoustic Guitar (nylon)",
        WaveformType.TRIANGLE,
        ADSREnvelope(0.01, 0.3, 0.4, 0.8),
        [(1.0, 1.0), (2.0, 0.3), (3.0, 0.1)],
    ),
    25: InstrumentProfile(
        "Acoustic Guitar (steel)",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.005, 0.4, 0.3, 0.9),
        [(1.0, 1.0), (2.0, 0.4), (4.0, 0.1)],
    ),
    26: InstrumentProfile(
        "Electric Guitar (jazz)",
        WaveformType.TRIANGLE,
        ADSREnvelope(0.01, 0.2, 0.7, 0.5),
    ),
    27: InstrumentProfile(
        "Electric Guitar (clean)", WaveformType.SINE, ADSREnvelope(0.01, 0.1, 0.8, 0.4)
    ),
    # Bass family (32-39)
    32: InstrumentProfile(
        "Acoustic Bass",
        WaveformType.SINE,
        ADSREnvelope(0.01, 0.2, 0.8, 0.6),
        [(1.0, 1.0), (2.0, 0.2)],
    ),
    33: InstrumentProfile(
        "Electric Bass (finger)",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.01, 0.1, 0.9, 0.4),
    ),
    34: InstrumentProfile(
        "Electric Bass (pick)", WaveformType.SQUARE, ADSREnvelope(0.005, 0.1, 0.7, 0.3)
    ),
    # Strings family (40-47)
    40: InstrumentProfile(
        "Violin",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.0, 1.0, 0.2),
        [(1.0, 1.0), (2.0, 0.3), (3.0, 0.1)],
        vibrato_rate=6.0,
        vibrato_depth=0.1,
    ),
    41: InstrumentProfile(
        "Viola",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.0, 1.0, 0.2),
        [(1.0, 1.0), (2.0, 0.2)],
        vibrato_rate=5.5,
        vibrato_depth=0.08,
    ),
    42: InstrumentProfile(
        "Cello",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.0, 1.0, 0.3),
        [(1.0, 1.0), (2.0, 0.3)],
        vibrato_rate=5.0,
        vibrato_depth=0.06,
    ),
    43: InstrumentProfile(
        "Contrabass",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.0, 1.0, 0.4),
        [(1.0, 1.0)],
        vibrato_rate=4.5,
        vibrato_depth=0.05,
    ),
    # Ensemble strings (48-55)
    48: InstrumentProfile(
        "String Ensemble 1",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.2, 0.0, 1.0, 0.3),
        [(1.0, 1.0), (2.0, 0.4), (3.0, 0.2)],
    ),
    49: InstrumentProfile(
        "String Ensemble 2",
        WaveformType.TRIANGLE,
        ADSREnvelope(0.3, 0.0, 1.0, 0.4),
        [(1.0, 1.0), (2.0, 0.3)],
    ),
    # Brass family (56-63)
    56: InstrumentProfile(
        "Trumpet",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.05, 0.1, 0.8, 0.2),
        [(1.0, 1.0), (2.0, 0.6), (3.0, 0.3), (4.0, 0.1)],
    ),
    57: InstrumentProfile(
        "Trombone",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.1, 0.9, 0.3),
        [(1.0, 1.0), (2.0, 0.5), (3.0, 0.2)],
    ),
    58: InstrumentProfile(
        "Tuba",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.1, 0.9, 0.4),
        [(1.0, 1.0), (2.0, 0.3)],
    ),
    59: InstrumentProfile(
        "Muted Trumpet",
        WaveformType.TRIANGLE,
        ADSREnvelope(0.05, 0.2, 0.6, 0.3),
        [(1.0, 1.0), (2.0, 0.4)],
    ),
    60: InstrumentProfile(
        "French Horn",
        WaveformType.SINE,
        ADSREnvelope(0.1, 0.0, 1.0, 0.4),
        [(1.0, 1.0), (2.0, 0.4), (3.0, 0.2)],
    ),
    61: InstrumentProfile(
        "Brass Section",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.1, 0.9, 0.3),
        [(1.0, 1.0), (2.0, 0.6), (3.0, 0.3)],
    ),
    # Reed family (64-71)
    64: InstrumentProfile(
        "Soprano Sax",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.05, 0.2, 0.7, 0.3),
        [(1.0, 1.0), (2.0, 0.5), (3.0, 0.2)],
        vibrato_rate=6.0,
        vibrato_depth=0.1,
    ),
    65: InstrumentProfile(
        "Alto Sax",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.05, 0.2, 0.7, 0.3),
        [(1.0, 1.0), (2.0, 0.4)],
        vibrato_rate=5.5,
        vibrato_depth=0.08,
    ),
    66: InstrumentProfile(
        "Tenor Sax",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.2, 0.8, 0.4),
        [(1.0, 1.0), (2.0, 0.3)],
        vibrato_rate=5.0,
        vibrato_depth=0.06,
    ),
    67: InstrumentProfile(
        "Baritone Sax",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.2, 0.8, 0.5),
        [(1.0, 1.0)],
        vibrato_rate=4.5,
        vibrato_depth=0.05,
    ),
    68: InstrumentProfile(
        "Oboe",
        WaveformType.TRIANGLE,
        ADSREnvelope(0.1, 0.1, 0.8, 0.3),
        [(1.0, 1.0), (2.0, 0.6), (3.0, 0.3)],
    ),
    69: InstrumentProfile(
        "English Horn",
        WaveformType.TRIANGLE,
        ADSREnvelope(0.1, 0.1, 0.9, 0.4),
        [(1.0, 1.0), (2.0, 0.4)],
    ),
    70: InstrumentProfile(
        "Bassoon",
        WaveformType.SAWTOOTH,
        ADSREnvelope(0.1, 0.1, 0.9, 0.5),
        [(1.0, 1.0), (2.0, 0.3)],
    ),
    71: InstrumentProfile(
        "Clarinet",
        WaveformType.SQUARE,
        ADSREnvelope(0.1, 0.1, 0.8, 0.3),
        [(1.0, 1.0), (3.0, 0.3), (5.0, 0.1)],
    ),
    # Pipe family (72-79)
    72: InstrumentProfile(
        "Piccolo",
        WaveformType.SINE,
        ADSREnvelope(0.01, 0.1, 0.8, 0.2),
        [(1.0, 1.0), (2.0, 0.3)],
    ),
    73: InstrumentProfile(
        "Flute",
        WaveformType.SINE,
        ADSREnvelope(0.1, 0.1, 0.8, 0.3),
        [(1.0, 1.0), (2.0, 0.2)],
    ),
    74: InstrumentProfile(
        "Recorder",
        WaveformType.TRIANGLE,
        ADSREnvelope(0.05, 0.1, 0.7, 0.3),
        [(1.0, 1.0)],
    ),
    75: InstrumentProfile(
        "Pan Flute", WaveformType.SINE, ADSREnvelope(0.1, 0.0, 1.0, 0.4), [(1.0, 1.0)]
    ),
}


def midi_note_to_frequency(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def generate_waveform(
    waveform_type: WaveformType, frequency: float, duration: float, sample_rate: int
) -> np.ndarray:
    """Generate a basic waveform."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)

    if waveform_type == WaveformType.SINE:
        return np.sin(2 * np.pi * frequency * t)
    elif waveform_type == WaveformType.SAWTOOTH:
        return 2 * (t * frequency - np.floor(t * frequency + 0.5))
    elif waveform_type == WaveformType.SQUARE:
        return np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform_type == WaveformType.TRIANGLE:
        return 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
    elif waveform_type == WaveformType.NOISE:
        return np.random.uniform(-1, 1, samples)
    else:
        return np.zeros(samples)


def apply_adsr_envelope(
    audio: np.ndarray, envelope: ADSREnvelope, sample_rate: int, note_duration: float
) -> np.ndarray:
    """Apply ADSR envelope to audio signal."""
    samples = len(audio)
    envelope_curve = np.ones(samples)

    # Convert times to samples
    attack_samples = int(envelope.attack * sample_rate)
    decay_samples = int(envelope.decay * sample_rate)
    release_samples = int(envelope.release * sample_rate)

    # Clamp to audio length
    attack_samples = min(attack_samples, samples // 4)
    decay_samples = min(decay_samples, samples // 4)
    release_samples = min(release_samples, samples // 2)

    # Attack phase
    if attack_samples > 0:
        envelope_curve[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay phase
    if decay_samples > 0 and attack_samples < samples:
        decay_end = min(attack_samples + decay_samples, samples)
        decay_range = decay_end - attack_samples
        if decay_range > 0:
            envelope_curve[attack_samples:decay_end] = np.linspace(
                1, envelope.sustain, decay_range
            )

    # Sustain phase (already set to sustain level or 1.0)
    sustain_start = attack_samples + decay_samples
    sustain_end = samples - release_samples
    if sustain_start < sustain_end:
        envelope_curve[sustain_start:sustain_end] = envelope.sustain

    # Release phase
    if release_samples > 0 and samples > release_samples:
        release_start = samples - release_samples
        envelope_curve[release_start:] = np.linspace(
            envelope.sustain, 0, release_samples
        )

    return audio * envelope_curve


class Voice:
    """Represents a single playing voice/note."""

    def __init__(
        self,
        note: int,
        velocity: int,
        instrument: InstrumentProfile,
        start_time: float,
        end_time: float,
        sample_rate: int,
    ):
        self.note = note
        self.velocity = velocity
        self.instrument = instrument
        self.start_time = start_time
        self.end_time = end_time
        self.sample_rate = sample_rate
        self.frequency = midi_note_to_frequency(note)
        self.amplitude = velocity / 127.0

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        """Render this voice for the given sample range."""
        start_time = start_sample / self.sample_rate
        duration = num_samples / self.sample_rate

        # Check if this voice is active during this time range
        if start_time >= self.end_time or start_time + duration <= self.start_time:
            return np.zeros(num_samples)

        # Calculate actual render duration within voice lifetime
        voice_start = max(start_time, self.start_time)
        voice_end = min(start_time + duration, self.end_time)
        voice_duration = voice_end - voice_start

        if voice_duration <= 0:
            return np.zeros(num_samples)

        # Generate base waveform
        audio = generate_waveform(
            self.instrument.waveform, self.frequency, voice_duration, self.sample_rate
        )

        # Add harmonics if specified
        if self.instrument.harmonics:
            harmonic_audio = np.zeros_like(audio)
            for freq_ratio, amplitude in self.instrument.harmonics:
                harmonic = (
                    generate_waveform(
                        self.instrument.waveform,
                        self.frequency * freq_ratio,
                        voice_duration,
                        self.sample_rate,
                    )
                    * amplitude
                )
                harmonic_audio += harmonic
            audio = harmonic_audio

        # Apply vibrato if specified
        if self.instrument.vibrato_rate > 0 and self.instrument.vibrato_depth > 0:
            t = np.linspace(0, voice_duration, len(audio), False)
            vibrato_freq = self.frequency * (
                2.0
                ** (
                    self.instrument.vibrato_depth
                    * np.sin(2 * np.pi * self.instrument.vibrato_rate * t)
                    / 12.0
                )
            )
            # Simple vibrato by frequency modulation
            vibrato_factor = vibrato_freq / self.frequency
            audio = audio * np.sin(2 * np.pi * self.frequency * t * vibrato_factor)

        # Apply ADSR envelope
        note_duration = self.end_time - self.start_time
        audio = apply_adsr_envelope(
            audio, self.instrument.envelope, self.sample_rate, note_duration
        )

        # Apply velocity scaling
        audio = audio * self.amplitude

        # Create output buffer and place audio at correct position
        output = np.zeros(num_samples)
        output_start = max(0, int((voice_start - start_time) * self.sample_rate))
        output_end = min(num_samples, output_start + len(audio))
        audio_start = max(0, int((start_time - voice_start) * self.sample_rate))
        audio_end = audio_start + (output_end - output_start)

        if output_end > output_start and audio_end > audio_start:
            output[output_start:output_end] = audio[audio_start:audio_end]

        return output


class PurePythonSynthesizer:
    """Pure Python MIDI synthesizer using numpy."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.voices: list[Voice] = []

    def add_note(
        self,
        note: int,
        velocity: int,
        start_time: float,
        end_time: float,
        program: int = 0,
    ):
        """Add a note to be synthesized."""
        # Get instrument profile (default to piano if not found)
        instrument = GM_INSTRUMENTS.get(program, GM_INSTRUMENTS[0])

        voice = Voice(
            note, velocity, instrument, start_time, end_time, self.sample_rate
        )
        self.voices.append(voice)

    def render(self, duration: float, chunk_size: int = 4096) -> np.ndarray:
        """Render all voices to audio."""
        total_samples = int(duration * self.sample_rate)
        output = np.zeros(total_samples, dtype=np.float32)

        # Render in chunks to manage memory
        for start_sample in range(0, total_samples, chunk_size):
            end_sample = min(start_sample + chunk_size, total_samples)
            chunk_samples = end_sample - start_sample

            chunk_audio = np.zeros(chunk_samples, dtype=np.float32)

            # Mix all active voices for this chunk
            for voice in self.voices:
                voice_audio = voice.render(start_sample, chunk_samples)
                chunk_audio += voice_audio

            output[start_sample:end_sample] = chunk_audio

        # Normalize to prevent clipping
        peak = np.abs(output).max()
        if peak > 0:
            output = output / peak * 0.8  # Leave some headroom

        return output

    def clear_voices(self):
        """Clear all voices."""
        self.voices.clear()


def synthesize_pretty_midi(
    midi_data, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """
    Synthesize a pretty_midi.PrettyMIDI object to audio.

    Args:
        midi_data: PrettyMIDI object
        sample_rate: Audio sample rate

    Returns:
        Synthesized audio as numpy array
    """
    synthesizer = PurePythonSynthesizer(sample_rate)

    # Extract notes from all instruments
    for instrument in midi_data.instruments:
        program = instrument.program if not instrument.is_drum else 128

        for note in instrument.notes:
            synthesizer.add_note(
                note.pitch, note.velocity, note.start, note.end, program
            )

    # Calculate total duration
    duration = midi_data.get_end_time()
    if duration <= 0:
        duration = 1.0  # Minimum duration

    # Add small buffer
    duration += 1.0

    # Render to audio
    audio = synthesizer.render(duration)

    # Convert to stereo
    if len(audio.shape) == 1:
        audio = np.column_stack([audio, audio])

    return audio
