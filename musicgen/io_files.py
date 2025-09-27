"""
File I/O operations for MusicGen.

This module handles:
- MIDI file discovery and loading
- MIDI file writing
- Audio file writing
- Directory management for outputs
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import mido
import numpy as np
import pretty_midi
from pydub import AudioSegment

from .config import ExportFormat, MusicGenConfig, AnalysisResult


def discover_midi_files(input_path: Path) -> List[Path]:
    """
    Discover MIDI files from input path.

    Args:
        input_path: File or directory path

    Returns:
        List of MIDI file paths

    Raises:
        ValueError: If no MIDI files found
    """
    midi_extensions = {".mid", ".midi"}
    midi_files = []

    if input_path.is_file():
        if input_path.suffix.lower() in midi_extensions:
            midi_files.append(input_path)
    elif input_path.is_dir():
        for ext in midi_extensions:
            midi_files.extend(input_path.glob(f"**/*{ext}"))

    if not midi_files:
        raise ValueError(f"No MIDI files found in {input_path}")

    return sorted(midi_files)


def load_midi_file(file_path: Path) -> pretty_midi.PrettyMIDI:
    """
    Load a MIDI file using pretty_midi.

    Args:
        file_path: Path to MIDI file

    Returns:
        PrettyMIDI object

    Raises:
        ValueError: If file cannot be loaded
    """
    try:
        return pretty_midi.PrettyMIDI(str(file_path))
    except Exception as e:
        raise ValueError(f"Failed to load MIDI file {file_path}: {e}")


def save_midi_file(
    midi_data: pretty_midi.PrettyMIDI,
    output_path: Path
) -> None:
    """
    Save a PrettyMIDI object to file.

    Args:
        midi_data: PrettyMIDI object to save
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_data.write(str(output_path))


def save_audio_file(
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: Path,
    bit_depth: int = 24
) -> None:
    """
    Save audio data to WAV file.

    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        output_path: Output file path
        bit_depth: Bit depth (16, 24, or 32)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize audio to prevent clipping
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Normalize to -1 to 1 range
    max_val = np.abs(audio_data).max()
    if max_val > 0:
        audio_data = audio_data / max_val * 0.99  # Slight headroom

    # Convert to integer format based on bit depth
    if bit_depth == 16:
        audio_int = (audio_data * 32767).astype(np.int16)
    elif bit_depth == 24:
        audio_int = (audio_data * 8388607).astype(np.int32)
    elif bit_depth == 32:
        audio_int = (audio_data * 2147483647).astype(np.int32)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    # Create AudioSegment and export
    if len(audio_int.shape) == 1:
        # Mono
        audio_segment = AudioSegment(
            audio_int.tobytes(),
            frame_rate=sample_rate,
            sample_width=bit_depth // 8,
            channels=1
        )
    else:
        # Stereo - interleave channels
        interleaved = np.empty((audio_int.shape[0] * 2,), dtype=audio_int.dtype)
        interleaved[0::2] = audio_int[:, 0]
        interleaved[1::2] = audio_int[:, 1]
        audio_segment = AudioSegment(
            interleaved.tobytes(),
            frame_rate=sample_rate,
            sample_width=bit_depth // 8,
            channels=2
        )

    audio_segment.export(str(output_path), format="wav")


def convert_wav_to_mp3(
    wav_path: Path,
    mp3_path: Path,
    bitrate: int = 192
) -> None:
    """
    Convert WAV file to MP3.

    Args:
        wav_path: Input WAV file path
        mp3_path: Output MP3 file path
        bitrate: MP3 bitrate in kbps
    """
    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_wav(str(wav_path))
    audio.export(
        str(mp3_path),
        format="mp3",
        bitrate=f"{bitrate}k"
    )


def save_analysis_json(
    analysis: AnalysisResult,
    output_path: Path
) -> None:
    """
    Save analysis results to JSON file.

    Args:
        analysis: Analysis results
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analysis_dict = analysis.dict()
    # Convert tuples to lists for JSON serialization
    analysis_dict["time_signature"] = list(analysis_dict["time_signature"])
    analysis_dict["sections"] = [
        list(section) for section in analysis_dict["sections"]
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_dict, f, indent=2, ensure_ascii=False)


def save_generation_report(
    config: MusicGenConfig,
    analysis: AnalysisResult,
    output_path: Path,
    rationale: str = ""
) -> None:
    """
    Save generation parameters and rationale to text file.

    Args:
        config: Configuration used for generation
        analysis: Analysis results
        output_path: Output text file path
        rationale: Additional rationale text
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("MusicGen Generation Report\n")
        f.write("=" * 30 + "\n\n")

        f.write("Input Analysis:\n")
        f.write(f"  Key: {analysis.key}\n")
        f.write(f"  Tempo: {analysis.tempo:.1f} BPM\n")
        f.write(f"  Time Signature: {analysis.time_signature[0]}/{analysis.time_signature[1]}\n")
        f.write(f"  Duration: {analysis.duration_seconds:.1f} seconds\n")
        f.write(f"  Note Density: {analysis.note_density:.2f} notes/second\n\n")

        f.write("Generation Parameters:\n")
        f.write(f"  Target Duration: {config.duration_seconds} seconds\n")
        f.write(f"  Instruments: {[inst.value for inst in config.instruments]}\n")
        f.write(f"  Voices: {config.voices}\n")
        f.write(f"  Style: {config.style}\n")
        f.write(f"  Key: {config.key.value if config.key else 'Auto-detected'}\n")
        f.write(f"  Tempo: {config.get_effective_tempo()}\n")
        f.write(f"  Seed: {config.seed}\n")
        f.write(f"  Export Formats: {[fmt.value for fmt in config.export_formats]}\n\n")

        if rationale:
            f.write("Generation Rationale:\n")
            f.write(rationale + "\n\n")

        f.write("Generated by MusicGen\n")


def create_output_directory(
    base_output_dir: Path,
    input_file_path: Path
) -> Path:
    """
    Create output directory with slug based on input filename.

    Args:
        base_output_dir: Base output directory
        input_file_path: Input file path for slug generation

    Returns:
        Created output directory path
    """
    # Create slug from input filename
    slug = input_file_path.stem.replace(" ", "_").lower()
    output_dir = base_output_dir / slug

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_output_files(
    output_dir: Path,
    config: MusicGenConfig,
    analysis: AnalysisResult,
    generated_midi: pretty_midi.PrettyMIDI,
    audio_data: Optional[np.ndarray] = None,
    rationale: str = ""
) -> dict:
    """
    Write all output files for a generation.

    Args:
        output_dir: Output directory
        config: Configuration used
        analysis: Analysis results
        generated_midi: Generated MIDI data
        audio_data: Generated audio data (if available)
        rationale: Generation rationale

    Returns:
        Dictionary of written file paths
    """
    written_files = {}

    # Always write analysis and report
    analysis_path = output_dir / "analysis.json"
    save_analysis_json(analysis, analysis_path)
    written_files["analysis"] = analysis_path

    report_path = output_dir / "report.txt"
    save_generation_report(config, analysis, report_path, rationale)
    written_files["report"] = report_path

    # Write files based on export formats
    if ExportFormat.MIDI in config.export_formats:
        midi_path = output_dir / "generated.mid"
        save_midi_file(generated_midi, midi_path)
        written_files["midi"] = midi_path

    if audio_data is not None:
        if ExportFormat.WAV in config.export_formats:
            wav_path = output_dir / "render.wav"
            save_audio_file(
                audio_data,
                config.sample_rate,
                wav_path,
                config.bit_depth
            )
            written_files["wav"] = wav_path

        if ExportFormat.MP3 in config.export_formats:
            mp3_path = output_dir / "render.mp3"
            # If WAV was also generated, convert from that
            if "wav" in written_files:
                convert_wav_to_mp3(
                    written_files["wav"],
                    mp3_path,
                    config.mp3_bitrate
                )
            else:
                # Create temporary WAV for conversion
                temp_wav = output_dir / "temp_render.wav"
                save_audio_file(
                    audio_data,
                    config.sample_rate,
                    temp_wav,
                    16  # Use 16-bit for MP3 source
                )
                convert_wav_to_mp3(temp_wav, mp3_path, config.mp3_bitrate)
                temp_wav.unlink()  # Remove temporary file

            written_files["mp3"] = mp3_path

    return written_files


def validate_soundfont(soundfont_path: Optional[Path]) -> bool:
    """
    Validate that a SoundFont file exists and is accessible.

    Args:
        soundfont_path: Path to SoundFont file

    Returns:
        True if valid, False otherwise
    """
    if soundfont_path is None:
        return False

    return (
        soundfont_path.exists() and
        soundfont_path.is_file() and
        soundfont_path.suffix.lower() == ".sf2"
    )