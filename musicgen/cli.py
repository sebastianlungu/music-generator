"""
Command-line interface for MusicGen.

This module provides the CLI entry point using Typer for
clean, user-friendly command-line interaction.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .config import (
    MusicGenConfig, ExportFormat, Instrument, MusicalKey
)
from .orchestration import (
    generate_arrangement, analyze_only, validate_configuration,
    get_generation_summary, batch_generate
)

# Set up CLI app
app = typer.Typer(
    name="musicgen",
    help="Generate musical arrangements from MIDI files",
    add_completion=False
)

# Set up console for rich output
console = Console()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def generate(
    input_path: Path = typer.Argument(
        ...,
        help="Input MIDI file or directory",
        exists=True
    ),
    output: Path = typer.Option(
        Path("./out"),
        "--output", "-o",
        help="Output directory"
    ),
    duration: int = typer.Option(
        120,
        "--duration-seconds", "-d",
        min=10, max=600,
        help="Maximum duration in seconds"
    ),
    instruments: str = typer.Option(
        "piano",
        "--instruments", "-i",
        help="Comma-separated list of instruments"
    ),
    voices: int = typer.Option(
        1,
        "--voices", "-v",
        min=1, max=8,
        help="Number of voices/parts"
    ),
    style: str = typer.Option(
        "classical",
        "--style", "-s",
        help="Musical style description"
    ),
    tempo_bpm: Optional[int] = typer.Option(
        None,
        "--tempo-bpm", "-t",
        min=60, max=200,
        help="Fixed tempo in BPM"
    ),
    tempo_range: Optional[str] = typer.Option(
        None,
        "--tempo-range", "-tr",
        help="Tempo range as 'min:max' (e.g., '90:120')"
    ),
    key: Optional[str] = typer.Option(
        None,
        "--key", "-k",
        help="Target musical key (e.g., 'C major', 'A minor')"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility"
    ),
    soundfont: Optional[Path] = typer.Option(
        None,
        "--soundfont", "-sf",
        help="Path to SoundFont file (.sf2)"
    ),
    export_formats: str = typer.Option(
        "midi,wav,mp3",
        "--export", "-e",
        help="Export formats (comma-separated): midi,wav,mp3"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-V",
        help="Enable verbose output"
    ),
    batch: bool = typer.Option(
        False,
        "--batch", "-b",
        help="Batch process all MIDI files in directory"
    )
) -> None:
    """Generate musical arrangements from MIDI files."""

    # Set up logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    try:
        # Parse instruments
        instrument_list = _parse_instruments(instruments)

        # Parse tempo range
        tempo_range_tuple = None
        if tempo_range:
            tempo_range_tuple = _parse_tempo_range(tempo_range)

        # Parse key
        musical_key = None
        if key:
            musical_key = _parse_musical_key(key)

        # Parse export formats
        export_format_list = _parse_export_formats(export_formats)

        # Create configuration
        config = MusicGenConfig(
            input_path=input_path,
            output_dir=output,
            duration_seconds=duration,
            instruments=instrument_list,
            voices=voices,
            style=style,
            tempo_bpm=tempo_bpm,
            tempo_range=tempo_range_tuple,
            key=musical_key,
            seed=seed,
            soundfont_path=soundfont,
            export_formats=export_format_list
        )

        # Validate configuration
        issues = validate_configuration(config)
        if issues:
            console.print("❌ Configuration issues found:", style="red bold")
            for issue in issues:
                console.print(f"  • {issue}", style="red")
            raise typer.Exit(1)

        # Run generation
        if batch and input_path.is_dir():
            _run_batch_generation(config)
        else:
            _run_single_generation(config)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red bold")
        logger.exception("Generation failed")
        raise typer.Exit(1)


@app.command()
def analyze(
    input_path: Path = typer.Argument(
        ...,
        help="Input MIDI file or directory",
        exists=True
    ),
    output_json: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save analysis to JSON file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    )
) -> None:
    """Analyze MIDI files without generating arrangements."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing MIDI file...", total=None)

            analysis_result = analyze_only(input_path)

            progress.update(task, description="Analysis complete ✓")

        # Display results
        _display_analysis_results(analysis_result)

        # Save to JSON if requested
        if output_json:
            analysis_dict = analysis_result.dict()
            with open(output_json, 'w') as f:
                json.dump(analysis_dict, f, indent=2)
            console.print(f"✅ Analysis saved to: {output_json}", style="green")

    except Exception as e:
        console.print(f"❌ Analysis failed: {e}", style="red bold")
        logger.exception("Analysis failed")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Display information about MusicGen and system capabilities."""

    from . import synthesis
    from . import __version__

    console.print(f"MusicGen v{__version__}", style="bold blue")
    console.print()

    # System capabilities
    table = Table(title="System Capabilities")
    table.add_column("Feature", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Notes")

    # Check FluidSynth
    if synthesis.check_fluidsynth_available():
        table.add_row("Audio Synthesis", "✅ Available", "FluidSynth found")
    else:
        table.add_row("Audio Synthesis", "❌ Unavailable", "Install pyfluidsynth")

    # Check dependencies
    try:
        import pretty_midi
        table.add_row("MIDI Processing", "✅ Available", f"pretty_midi {pretty_midi.__version__}")
    except ImportError:
        table.add_row("MIDI Processing", "❌ Unavailable", "Install pretty_midi")

    try:
        import music21
        table.add_row("Music Analysis", "✅ Available", "music21 found")
    except ImportError:
        table.add_row("Music Analysis", "❌ Unavailable", "Install music21")

    try:
        import pydub
        table.add_row("Audio Conversion", "✅ Available", "pydub found")
    except ImportError:
        table.add_row("Audio Conversion", "❌ Unavailable", "Install pydub")

    console.print(table)
    console.print()

    # Supported formats
    console.print("Supported formats:", style="bold")
    console.print("  Input: MIDI (.mid, .midi)")
    console.print("  Output: MIDI (.mid), WAV (.wav), MP3 (.mp3)")
    console.print()

    # Available instruments
    console.print("Available instruments:", style="bold")
    instruments = [inst.value for inst in Instrument]
    console.print(f"  {', '.join(instruments)}")


def _parse_instruments(instruments_str: str) -> List[Instrument]:
    """Parse comma-separated instrument names."""
    instrument_names = [name.strip().lower() for name in instruments_str.split(",")]
    instruments = []

    for name in instrument_names:
        try:
            # Try to find matching instrument
            for instrument in Instrument:
                if instrument.value.lower() == name or instrument.name.lower() == name:
                    instruments.append(instrument)
                    break
            else:
                # No exact match found
                console.print(f"⚠️  Unknown instrument: {name}, using piano instead", style="yellow")
                instruments.append(Instrument.PIANO)
        except Exception:
            console.print(f"⚠️  Invalid instrument: {name}, using piano instead", style="yellow")
            instruments.append(Instrument.PIANO)

    return instruments


def _parse_tempo_range(tempo_range_str: str) -> Tuple[int, int]:
    """Parse tempo range string like '90:120'."""
    try:
        parts = tempo_range_str.split(":")
        if len(parts) != 2:
            raise ValueError("Tempo range must be in format 'min:max'")

        min_tempo = int(parts[0])
        max_tempo = int(parts[1])

        if min_tempo >= max_tempo:
            raise ValueError("Minimum tempo must be less than maximum")

        if min_tempo < 60 or max_tempo > 200:
            raise ValueError("Tempo values must be between 60 and 200 BPM")

        return (min_tempo, max_tempo)

    except ValueError as e:
        raise typer.BadParameter(f"Invalid tempo range: {e}")


def _parse_musical_key(key_str: str) -> MusicalKey:
    """Parse musical key string."""
    try:
        # Try to find matching key
        for musical_key in MusicalKey:
            if musical_key.value.lower() == key_str.lower():
                return musical_key

        raise ValueError(f"Unknown key: {key_str}")

    except ValueError as e:
        raise typer.BadParameter(str(e))


def _parse_export_formats(formats_str: str) -> List[ExportFormat]:
    """Parse comma-separated export formats."""
    format_names = [name.strip().lower() for name in formats_str.split(",")]
    formats = []

    for name in format_names:
        try:
            formats.append(ExportFormat(name))
        except ValueError:
            console.print(f"⚠️  Unknown format: {name}, ignoring", style="yellow")

    if not formats:
        console.print("⚠️  No valid formats specified, using MIDI", style="yellow")
        formats = [ExportFormat.MIDI]

    return formats


def _run_single_generation(config: MusicGenConfig) -> None:
    """Run generation for a single file or first file in directory."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating arrangement...", total=None)

        result = generate_arrangement(config)

        if result.success:
            progress.update(task, description="Generation complete ✓")
        else:
            progress.update(task, description="Generation failed ❌")

    if result.success:
        summary = get_generation_summary(result)
        _display_generation_summary(summary)
    else:
        console.print(f"❌ Generation failed: {result.error_message}", style="red bold")
        raise typer.Exit(1)


def _run_batch_generation(config: MusicGenConfig) -> None:
    """Run batch generation for multiple files."""

    console.print(f"🎵 Running batch generation from: {config.input_path}", style="blue")

    results = batch_generate(
        config.input_path,
        config.output_dir,
        config
    )

    # Display summary
    successful = sum(1 for r in results if r.success)
    total = len(results)

    console.print()
    console.print(f"📊 Batch generation complete: {successful}/{total} files processed", style="bold")

    if successful < total:
        console.print("❌ Failed files:", style="red")
        for result in results:
            if not result.success:
                input_file = result.config.input_path.name
                console.print(f"  • {input_file}: {result.error_message}", style="red")


def _display_analysis_results(analysis: "AnalysisResult") -> None:
    """Display analysis results in a formatted table."""

    table = Table(title="MIDI Analysis Results")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Key", analysis.key)
    table.add_row("Tempo", f"{analysis.tempo:.1f} BPM")
    table.add_row("Time Signature", f"{analysis.time_signature[0]}/{analysis.time_signature[1]}")
    table.add_row("Duration", f"{analysis.duration_seconds:.1f} seconds")
    table.add_row("Note Density", f"{analysis.note_density:.2f} notes/second")
    table.add_row("Sections", str(len(analysis.sections)))
    table.add_row("Instruments", str(len(analysis.instrument_programs)))

    console.print(table)

    # Show pitch histogram if verbose
    if any(analysis.pitch_histogram):
        console.print("\nPitch Class Distribution:", style="bold")
        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for i, value in enumerate(analysis.pitch_histogram):
            if value > 0.01:  # Only show significant values
                console.print(f"  {pitch_names[i]}: {value:.2%}")


def _display_generation_summary(summary: dict) -> None:
    """Display generation summary."""

    console.print("🎵 Generation completed successfully!", style="green bold")
    console.print()

    # Input analysis
    analysis = summary["input_analysis"]
    table = Table(title="Input Analysis")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Key", analysis["key"])
    table.add_row("Tempo", f"{analysis['tempo']} BPM")
    table.add_row("Time Signature", analysis["time_signature"])
    table.add_row("Duration", f"{analysis['duration']} seconds")

    console.print(table)
    console.print()

    # Generated files
    files_table = Table(title="Generated Files")
    files_table.add_column("Type", style="cyan")
    files_table.add_column("Status", style="green")
    files_table.add_column("Size", style="white")

    for file_type, file_info in summary["files"].items():
        status = "✅ Created" if file_info["exists"] else "❌ Failed"
        size = f"{file_info['size_bytes']:,} bytes" if file_info["exists"] else "N/A"
        files_table.add_row(file_type.upper(), status, size)

    console.print(files_table)
    console.print()

    # Show file paths
    console.print("📁 Output files:", style="bold")
    for file_type, file_info in summary["files"].items():
        if file_info["exists"]:
            console.print(f"  {file_type}: {file_info['path']}")


def main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n👋 Cancelled by user", style="yellow")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n💥 Unexpected error: {e}", style="red bold")
        logger.exception("Unexpected error in CLI")
        sys.exit(1)


if __name__ == "__main__":
    main()