"""
Configuration models for MusicGen using Pydantic.

This module defines all configuration classes for the MusicGen application,
including CLI parameters, analysis settings, and generation options.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class ExportFormat(str, Enum):
    """Available export formats."""
    MIDI = "midi"
    WAV = "wav"
    MP3 = "mp3"


class Instrument(str, Enum):
    """Standard instrument names mapped to General MIDI."""
    PIANO = "piano"
    GUITAR = "guitar"
    VIOLIN = "violin"
    CELLO = "cello"
    FLUTE = "flute"
    CLARINET = "clarinet"
    TRUMPET = "trumpet"
    SAXOPHONE = "saxophone"
    VOICE = "voice"
    CHOIR = "choir"
    STRINGS = "string_ensemble"
    BRASS = "brass_ensemble"
    WOODWINDS = "woodwind_ensemble"
    ORGAN = "organ"
    BASS = "bass"
    DRUMS = "drums"


class MusicalKey(str, Enum):
    """Musical keys for arrangement generation."""
    C_MAJOR = "C major"
    G_MAJOR = "G major"
    D_MAJOR = "D major"
    A_MAJOR = "A major"
    E_MAJOR = "E major"
    B_MAJOR = "B major"
    F_SHARP_MAJOR = "F# major"
    C_SHARP_MAJOR = "C# major"
    F_MAJOR = "F major"
    B_FLAT_MAJOR = "Bb major"
    E_FLAT_MAJOR = "Eb major"
    A_FLAT_MAJOR = "Ab major"
    D_FLAT_MAJOR = "Db major"
    G_FLAT_MAJOR = "Gb major"
    C_FLAT_MAJOR = "Cb major"
    A_MINOR = "A minor"
    E_MINOR = "E minor"
    B_MINOR = "B minor"
    F_SHARP_MINOR = "F# minor"
    C_SHARP_MINOR = "C# minor"
    G_SHARP_MINOR = "G# minor"
    D_SHARP_MINOR = "D# minor"
    A_SHARP_MINOR = "A# minor"
    D_MINOR = "D minor"
    G_MINOR = "G minor"
    C_MINOR = "C minor"
    F_MINOR = "F minor"
    B_FLAT_MINOR = "Bb minor"
    E_FLAT_MINOR = "Eb minor"
    A_FLAT_MINOR = "Ab minor"


class AnalysisResult(BaseModel):
    """Results from MIDI analysis."""
    key: str
    tempo: float
    time_signature: Tuple[int, int]
    duration_seconds: float
    pitch_histogram: List[float]
    note_density: float
    sections: List[Tuple[float, float]]  # (start_time, end_time) pairs
    instrument_programs: List[int]


class MusicGenConfig(BaseModel):
    """Main configuration for MusicGen operations."""

    # Input/Output
    input_path: Path = Field(..., description="Input MIDI file or directory")
    output_dir: Path = Field(
        default=Path("./out"),
        description="Output directory for generated files"
    )

    # Generation parameters
    duration_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Maximum duration in seconds"
    )
    instruments: List[Instrument] = Field(
        default=[Instrument.PIANO],
        min_items=1,
        max_items=8,
        description="Instruments to use in arrangement"
    )
    voices: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of voices/parts"
    )
    style: str = Field(
        default="classical",
        description="Musical style description"
    )

    # Musical parameters
    tempo_bpm: Optional[int] = Field(
        default=None,
        ge=60,
        le=200,
        description="Fixed tempo in BPM"
    )
    tempo_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Tempo range (min, max) in BPM"
    )
    key: Optional[MusicalKey] = Field(
        default=None,
        description="Target key (auto-detect if None)"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    # Audio rendering
    soundfont_path: Optional[Path] = Field(
        default=None,
        description="Path to SoundFont file (.sf2)"
    )
    export_formats: List[ExportFormat] = Field(
        default=[ExportFormat.MIDI, ExportFormat.WAV, ExportFormat.MP3],
        description="Output formats to generate"
    )

    # Audio quality settings
    sample_rate: int = Field(default=44100, description="Audio sample rate")
    bit_depth: int = Field(default=24, description="Audio bit depth")
    mp3_bitrate: int = Field(default=192, description="MP3 bitrate in kbps")

    @validator("tempo_range")
    def validate_tempo_range(cls, v):
        """Ensure tempo range is valid."""
        if v is not None:
            min_tempo, max_tempo = v
            if min_tempo >= max_tempo:
                raise ValueError("min_tempo must be less than max_tempo")
            if min_tempo < 60 or max_tempo > 200:
                raise ValueError("Tempo values must be between 60 and 200 BPM")
        return v

    @validator("tempo_bpm", "tempo_range")
    def validate_tempo_exclusivity(cls, v, values):
        """Ensure only one tempo parameter is set."""
        if v is not None and values.get("tempo_bpm") is not None:
            raise ValueError("Cannot set both tempo_bpm and tempo_range")
        return v

    @validator("input_path")
    def validate_input_path(cls, v):
        """Ensure input path exists."""
        if not v.exists():
            raise ValueError(f"Input path does not exist: {v}")
        return v

    @validator("soundfont_path")
    def validate_soundfont_path(cls, v):
        """Ensure SoundFont path exists if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"SoundFont file does not exist: {v}")
        return v

    def get_effective_tempo(self) -> Union[int, Tuple[int, int]]:
        """Get the effective tempo setting."""
        if self.tempo_bpm is not None:
            return self.tempo_bpm
        if self.tempo_range is not None:
            return self.tempo_range
        return 100  # Default tempo


class ArrangementConfig(BaseModel):
    """Configuration specific to arrangement generation."""

    style_rules: dict = Field(
        default_factory=lambda: {
            "classical": {
                "voice_leading": True,
                "harmonic_rhythm": "moderate",
                "texture": "polyphonic"
            },
            "ambient": {
                "sustained_chords": True,
                "sparse_melody": True,
                "texture": "atmospheric"
            },
            "rock": {
                "drums": True,
                "power_chords": True,
                "texture": "rhythmic"
            },
            "jazz": {
                "extended_harmony": True,
                "syncopation": True,
                "texture": "interactive"
            }
        }
    )

    humanization_amount: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Amount of timing/velocity humanization"
    )

    voice_leading_strictness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How strictly to follow voice leading rules"
    )


class WebUIConfig(BaseModel):
    """Configuration for the web UI."""

    host: str = Field(default="127.0.0.1", description="Host address")
    port: int = Field(default=7860, ge=1024, le=65535, description="Port number")
    share: bool = Field(default=False, description="Create public shareable link")
    debug: bool = Field(default=False, description="Enable debug mode")
    upload_max_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        description="Maximum upload file size in bytes"
    )