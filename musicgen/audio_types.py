"""
Shared audio types and interfaces for MusicGen.

This module defines common types, exceptions, and interfaces used throughout
the audio processing pipeline. It provides a central location for audio-related
constants and capability detection.
"""

import importlib.util
import os
import sys
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np


class AudioCapability(str, Enum):
    """Available audio processing capabilities."""

    MIDI_ONLY = "midi_only"
    AUDIO_SYNTHESIS = "audio_synthesis"  # FluidSynth available
    AUDIO_CONVERSION = "audio_conversion"  # FFmpeg available
    FULL_AUDIO = "full_audio"  # Both synthesis and conversion


class AudioFormat(str, Enum):
    """Supported audio formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class AudioError(Exception):
    """Base exception for audio-related errors."""

    pass


class SynthesisError(AudioError):
    """Exception raised during audio synthesis."""

    pass


class ConversionError(AudioError):
    """Exception raised during audio format conversion."""

    pass


class DependencyError(AudioError):
    """Exception raised when required dependencies are missing."""

    pass


class FluidSynthError(SynthesisError):
    """Exception raised for FluidSynth-specific issues."""

    pass


class FFmpegError(ConversionError):
    """Exception raised for FFmpeg-specific issues."""

    pass


class AudioInfo:
    """Information about audio data."""

    def __init__(
        self,
        duration: float,
        channels: int,
        samples: int,
        sample_rate: int,
        peak_level: float = 0.0,
        rms_level: float = 0.0,
    ):
        self.duration = duration
        self.channels = channels
        self.samples = samples
        self.sample_rate = sample_rate
        self.peak_level = peak_level
        self.rms_level = rms_level

    @property
    def peak_db(self) -> float:
        """Peak level in dB."""
        return 20 * np.log10(self.peak_level) if self.peak_level > 0 else -np.inf

    @property
    def rms_db(self) -> float:
        """RMS level in dB."""
        return 20 * np.log10(self.rms_level) if self.rms_level > 0 else -np.inf

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "duration": self.duration,
            "channels": self.channels,
            "samples": self.samples,
            "sample_rate": self.sample_rate,
            "peak_level": self.peak_level,
            "rms_level": self.rms_level,
            "peak_db": self.peak_db,
            "rms_db": self.rms_db,
        }


class AudioProcessor(Protocol):
    """Protocol for audio processing classes."""

    def process(self, audio_data: np.ndarray, **kwargs) -> np.ndarray:
        """Process audio data."""
        ...


class AudioSynthesizer(ABC):
    """Abstract base class for audio synthesizers."""

    @abstractmethod
    def synthesize(
        self,
        midi_data,  # pretty_midi.PrettyMIDI (avoid undefined name)
        **kwargs,
    ) -> np.ndarray:
        """Synthesize MIDI data to audio."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if synthesizer is available."""
        pass


class AudioConverter(ABC):
    """Abstract base class for audio converters."""

    @abstractmethod
    def convert(self, input_path: Path, output_path: Path, **kwargs) -> None:
        """Convert audio between formats."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if converter is available."""
        pass


# Audio processing constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = 24
DEFAULT_MP3_BITRATE = 192
DEFAULT_CHANNELS = 2

# Audio quality presets
AUDIO_QUALITY_PRESETS = {
    "low": {"sample_rate": 22050, "bit_depth": 16, "mp3_bitrate": 128},
    "medium": {"sample_rate": 44100, "bit_depth": 16, "mp3_bitrate": 192},
    "high": {"sample_rate": 44100, "bit_depth": 24, "mp3_bitrate": 256},
    "ultra": {"sample_rate": 96000, "bit_depth": 32, "mp3_bitrate": 320},
}

# Common file paths and environment variables
FLUIDSYNTH_ENV_VARS = ["FLUIDSYNTH_ROOT", "FLUIDSYNTH_PATH", "FLUIDSYNTH_LIB"]

FFMPEG_ENV_VARS = ["FFMPEG_ROOT", "FFMPEG_PATH", "FFMPEG_BINARY"]

# Windows-specific paths to search for FluidSynth
WINDOWS_FLUIDSYNTH_PATHS = [
    "C:/Program Files/FluidSynth",
    "C:/Program Files (x86)/FluidSynth",
    "C:/FluidSynth",
    "C:/Tools/FluidSynth",
]

# Common SoundFont locations
COMMON_SOUNDFONT_PATHS = [
    "/usr/share/soundfonts",
    "/usr/local/share/soundfonts",
    "C:/soundfonts",
    "C:/Windows/System32/drivers/gm.dls",  # Windows GM set
]


def detect_fluidsynth_path() -> Path | None:
    """
    Detect FluidSynth installation path.

    Returns:
        Path to FluidSynth if found, None otherwise
    """
    # Check environment variables first
    for env_var in FLUIDSYNTH_ENV_VARS:
        if env_var in os.environ:
            path = Path(os.environ[env_var])
            if path.exists():
                return path

    # Platform-specific search
    if sys.platform == "win32":
        for search_path in WINDOWS_FLUIDSYNTH_PATHS:
            path = Path(search_path)
            if path.exists():
                return path

    return None


def detect_ffmpeg_path() -> Path | None:
    """
    Detect FFmpeg installation path.

    Returns:
        Path to FFmpeg if found, None otherwise
    """
    # Check environment variables first
    for env_var in FFMPEG_ENV_VARS:
        if env_var in os.environ:
            path = Path(os.environ[env_var])
            if path.exists():
                return path

    # Try to find in PATH
    import shutil

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return Path(ffmpeg_path).parent

    return None


def get_audio_capabilities() -> list[AudioCapability]:
    """
    Detect available audio processing capabilities.

    Returns:
        List of available capabilities
    """
    capabilities = [AudioCapability.MIDI_ONLY]  # Always available

    # Check Pure Python synthesis availability (preferred)
    if all(
        importlib.util.find_spec(mod)
        for mod in ["numpy", "pretty_midi", "scipy", "soundfile"]
    ):
        capabilities.append(AudioCapability.AUDIO_SYNTHESIS)
    # Fallback: Check FluidSynth availability
    elif all(importlib.util.find_spec(mod) for mod in ["fluidsynth", "pretty_midi"]):
        capabilities.append(AudioCapability.AUDIO_SYNTHESIS)
    # Basic fallback: Check mido + numpy availability (minimal synthesis)
    elif all(importlib.util.find_spec(mod) for mod in ["mido", "numpy"]):
        capabilities.append(AudioCapability.AUDIO_SYNTHESIS)

    # Check FFmpeg availability for format conversion
    if importlib.util.find_spec("pydub") and detect_ffmpeg_path() is not None:
        capabilities.append(AudioCapability.AUDIO_CONVERSION)

    # Check if full audio pipeline is available
    if (
        AudioCapability.AUDIO_SYNTHESIS in capabilities
        and AudioCapability.AUDIO_CONVERSION in capabilities
    ):
        capabilities.append(AudioCapability.FULL_AUDIO)

    return capabilities


def is_capability_available(capability: AudioCapability) -> bool:
    """
    Check if a specific capability is available.

    Args:
        capability: Capability to check

    Returns:
        True if capability is available
    """
    return capability in get_audio_capabilities()


def get_missing_dependencies(capability: AudioCapability) -> list[str]:
    """
    Get list of missing dependencies for a capability.

    Args:
        capability: Capability to check

    Returns:
        List of missing dependency names
    """
    missing = []

    if capability in [AudioCapability.AUDIO_SYNTHESIS, AudioCapability.FULL_AUDIO]:
        if not importlib.util.find_spec("fluidsynth"):
            missing.append("pyfluidsynth")

        if not importlib.util.find_spec("pretty_midi"):
            missing.append("pretty_midi")

    if capability in [AudioCapability.AUDIO_CONVERSION, AudioCapability.FULL_AUDIO]:
        if not importlib.util.find_spec("pydub"):
            missing.append("pydub")

        if detect_ffmpeg_path() is None:
            missing.append("ffmpeg")

    return missing


def create_installation_instructions(capability: AudioCapability) -> str:
    """
    Create installation instructions for missing dependencies.

    Args:
        capability: Capability to get instructions for

    Returns:
        Installation instructions as string
    """
    missing = get_missing_dependencies(capability)
    if not missing:
        return "All dependencies are available."

    instructions = ["Missing dependencies found. Install with:"]

    python_packages = [
        dep for dep in missing if dep in ["pyfluidsynth", "pretty_midi", "pydub"]
    ]
    system_packages = [dep for dep in missing if dep in ["ffmpeg"]]

    if python_packages:
        instructions.append(f"  uv add {' '.join(python_packages)}")

    if "ffmpeg" in system_packages:
        if sys.platform == "win32":
            instructions.append(
                "  # For Windows: Download FFmpeg from https://ffmpeg.org/download.html"
            )
            instructions.append(
                "  # Extract and add to PATH, or set FFMPEG_PATH environment variable"
            )
        elif sys.platform == "darwin":
            instructions.append("  # For macOS: brew install ffmpeg")
        else:
            instructions.append(
                "  # For Linux: sudo apt install ffmpeg  # or equivalent for your distro"
            )

    if "pyfluidsynth" in python_packages:
        instructions.append("")
        instructions.append("Additional FluidSynth setup:")
        if sys.platform == "win32":
            instructions.append(
                "  # Windows: Download FluidSynth from https://www.fluidsynth.org/"
            )
            instructions.append(
                "  # Set FLUIDSYNTH_ROOT environment variable to installation path"
            )
        else:
            instructions.append("  # Linux/macOS: Install FluidSynth system package")
            instructions.append(
                "  # Linux: sudo apt install fluidsynth libfluidsynth-dev"
            )
            instructions.append("  # macOS: brew install fluidsynth")

    return "\n".join(instructions)


def warn_missing_capability(capability: AudioCapability, context: str = "") -> None:
    """
    Issue a warning about missing capability.

    Args:
        capability: Missing capability
        context: Additional context for the warning
    """
    if is_capability_available(capability):
        return  # No warning needed

    context_str = f" ({context})" if context else ""
    instructions = create_installation_instructions(capability)

    warnings.warn(
        f"Audio capability '{capability.value}' not available{context_str}.\n"
        f"{instructions}",
        UserWarning,
        stacklevel=2,
    )


def require_capability(capability: AudioCapability, context: str = "") -> None:
    """
    Require a specific capability and raise DependencyError if not available.

    Args:
        capability: Required capability
        context: Additional context for the error

    Raises:
        DependencyError: If capability is not available
    """
    if is_capability_available(capability):
        return  # Capability is available

    context_str = f" for {context}" if context else ""
    instructions = create_installation_instructions(capability)

    raise DependencyError(
        f"Required audio capability '{capability.value}' not available{context_str}.\n\n"
        f"{instructions}\n\n"
        f"Please install the required dependencies or use MIDI-only mode."
    )


def validate_export_formats(export_formats: list[str]) -> list[str]:
    """
    Validate export formats against available capabilities.

    Args:
        export_formats: List of requested export formats

    Returns:
        List of validated export formats

    Raises:
        DependencyError: If any requested format requires unavailable capabilities
    """
    validated_formats = []

    for fmt in export_formats:
        fmt_lower = fmt.lower()

        if fmt_lower == "midi":
            validated_formats.append(fmt_lower)
        elif fmt_lower == "wav":
            require_capability(
                AudioCapability.AUDIO_SYNTHESIS, f"WAV export (requested format: {fmt})"
            )
            validated_formats.append(fmt_lower)
        elif fmt_lower == "mp3":
            require_capability(
                AudioCapability.FULL_AUDIO, f"MP3 export (requested format: {fmt})"
            )
            validated_formats.append(fmt_lower)
        else:
            # For other formats, require full audio pipeline
            require_capability(
                AudioCapability.FULL_AUDIO, f"audio export (requested format: {fmt})"
            )
            validated_formats.append(fmt_lower)

    return validated_formats


def get_installation_mode() -> str:
    """
    Get the current installation mode based on available capabilities.

    Returns:
        Installation mode string
    """
    capabilities = get_audio_capabilities()

    if AudioCapability.FULL_AUDIO in capabilities:
        return "full-audio"
    elif AudioCapability.AUDIO_SYNTHESIS in capabilities:
        return "audio-synthesis"
    elif AudioCapability.AUDIO_CONVERSION in capabilities:
        return "audio-conversion"
    else:
        return "midi-only"


def create_upgrade_instructions() -> str:
    """
    Create instructions for upgrading from minimal to full installation.

    Returns:
        Upgrade instructions as string
    """
    current_mode = get_installation_mode()

    if current_mode == "full-audio":
        return "✅ Full audio capabilities available."

    instructions = [
        f"📦 Current installation mode: {current_mode}",
        "",
        "🚀 To enable audio features, install additional dependencies:",
        "",
    ]

    if current_mode == "midi-only":
        instructions.extend(
            [
                "# For basic audio synthesis (WAV export):",
                "uv sync --extra audio-synthesis",
                "",
                "# For full audio pipeline (MP3 export):",
                "uv sync --extra all",
                "",
            ]
        )
    elif current_mode in ["audio-synthesis", "audio-conversion"]:
        instructions.extend(
            [
                "# For full audio pipeline (MP3 export):",
                "uv sync --extra all",
                "",
            ]
        )

    instructions.extend(
        [
            "📚 Installation options:",
            "  uv sync                     # MIDI analysis and arrangement only",
            "  uv sync --extra web-ui      # Add web interface",
            "  uv sync --extra audio-synthesis  # Add WAV audio synthesis",
            "  uv sync --extra all         # Full features (audio + web UI)",
        ]
    )

    return "\n".join(instructions)
