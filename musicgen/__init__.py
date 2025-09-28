"""
MusicGen: A typed, testable Python tool for MIDI analysis and arrangement generation.

This package provides functionality to:
- Analyze MIDI files (key, tempo, structure, etc.)
- Generate new musical arrangements
- Render audio using SoundFonts
- Export to MIDI, WAV, and MP3 formats
"""

__version__ = "0.1.0"
__author__ = "MusicGen Team"

from .config import MusicGenConfig
from .orchestration import generate_arrangement

__all__ = [
    "MusicGenConfig",
    "generate_arrangement",
]
