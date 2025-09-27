"""
Web UI for MusicGen using Gradio.

This module provides a user-friendly web interface for:
- File upload (MIDI and SoundFont)
- Parameter configuration
- Real-time generation
- Audio preview and download
"""

import json
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any

import gradio as gr
import pandas as pd

from .config import (
    MusicGenConfig, ExportFormat, Instrument, MusicalKey, WebUIConfig
)
from .orchestration import generate_arrangement, get_generation_summary
from . import synthesis

# Set up logging
logger = logging.getLogger(__name__)


class WebUIState:
    """Manage state for the web UI."""

    def __init__(self):
        self.current_result = None
        self.temp_files = []

    def clear_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        self.temp_files.clear()

    def add_temp_file(self, file_path: Path):
        """Add a temporary file to track for cleanup."""
        self.temp_files.append(file_path)


# Global state instance
ui_state = WebUIState()


def create_interface(config: Optional[WebUIConfig] = None) -> gr.Blocks:
    """
    Create the Gradio interface.

    Args:
        config: Web UI configuration

    Returns:
        Gradio Blocks interface
    """
    if config is None:
        config = WebUIConfig()

    with gr.Blocks(
        title="MusicGen - Musical Arrangement Generator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .analysis-table {
            font-family: monospace;
        }
        """
    ) as interface:

        gr.Markdown(
            """
            # 🎵 MusicGen

            Generate musical arrangements from MIDI files using AI-powered analysis and rule-based composition.

            **Steps:**
            1. Upload a MIDI file
            2. Optionally upload a SoundFont for audio synthesis
            3. Configure generation parameters
            4. Click "Generate Arrangement"
            5. Preview and download results
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # File upload section
                gr.Markdown("## 📁 File Upload")

                midi_file = gr.File(
                    label="MIDI File",
                    file_types=[".mid", ".midi"],
                    file_count="single"
                )

                soundfont_file = gr.File(
                    label="SoundFont File (Optional)",
                    file_types=[".sf2"],
                    file_count="single"
                )

                # Generation parameters
                gr.Markdown("## ⚙️ Generation Parameters")

                duration = gr.Slider(
                    minimum=10,
                    maximum=300,
                    value=120,
                    step=10,
                    label="Duration (seconds)"
                )

                instruments = gr.CheckboxGroup(
                    choices=[inst.value for inst in Instrument],
                    value=["piano"],
                    label="Instruments"
                )

                voices = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=1,
                    label="Number of Voices"
                )

                style = gr.Textbox(
                    value="classical",
                    label="Musical Style",
                    placeholder="e.g., classical, jazz, rock, ambient"
                )

                # Musical parameters
                gr.Markdown("### 🎼 Musical Parameters")

                with gr.Row():
                    tempo_mode = gr.Radio(
                        choices=["auto", "fixed", "range"],
                        value="auto",
                        label="Tempo Mode"
                    )

                tempo_bpm = gr.Slider(
                    minimum=60,
                    maximum=200,
                    value=120,
                    step=1,
                    label="Fixed Tempo (BPM)",
                    visible=False
                )

                with gr.Row():
                    tempo_min = gr.Slider(
                        minimum=60,
                        maximum=200,
                        value=90,
                        step=1,
                        label="Min Tempo",
                        visible=False
                    )
                    tempo_max = gr.Slider(
                        minimum=60,
                        maximum=200,
                        value=120,
                        step=1,
                        label="Max Tempo",
                        visible=False
                    )

                key_mode = gr.Radio(
                    choices=["auto", "specified"],
                    value="auto",
                    label="Key Mode"
                )

                musical_key = gr.Dropdown(
                    choices=[key.value for key in MusicalKey],
                    value="C major",
                    label="Musical Key",
                    visible=False
                )

                seed = gr.Number(
                    value=42,
                    label="Random Seed",
                    precision=0
                )

                # Export options
                gr.Markdown("### 📤 Export Options")

                export_formats = gr.CheckboxGroup(
                    choices=[fmt.value for fmt in ExportFormat],
                    value=["midi", "wav", "mp3"],
                    label="Export Formats"
                )

                # Generate button
                generate_btn = gr.Button(
                    "🎵 Generate Arrangement",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                # Results section
                gr.Markdown("## 📊 Results")

                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to generate...",
                    interactive=False
                )

                # Analysis results
                analysis_table = gr.DataFrame(
                    label="Input Analysis",
                    headers=["Property", "Value"],
                    datatype=["str", "str"],
                    elem_classes=["analysis-table"]
                )

                # Audio preview
                audio_output = gr.Audio(
                    label="Generated Audio Preview",
                    visible=False
                )

                # Download links
                gr.Markdown("### 📥 Download Files")

                download_midi = gr.File(
                    label="MIDI File",
                    visible=False
                )

                download_wav = gr.File(
                    label="WAV Audio",
                    visible=False
                )

                download_mp3 = gr.File(
                    label="MP3 Audio",
                    visible=False
                )

                download_analysis = gr.File(
                    label="Analysis JSON",
                    visible=False
                )

                download_report = gr.File(
                    label="Generation Report",
                    visible=False
                )

        # Event handlers
        def update_tempo_controls(mode):
            """Update tempo control visibility based on mode."""
            return {
                tempo_bpm: gr.update(visible=(mode == "fixed")),
                tempo_min: gr.update(visible=(mode == "range")),
                tempo_max: gr.update(visible=(mode == "range"))
            }

        def update_key_controls(mode):
            """Update key control visibility based on mode."""
            return {
                musical_key: gr.update(visible=(mode == "specified"))
            }

        tempo_mode.change(
            update_tempo_controls,
            inputs=[tempo_mode],
            outputs=[tempo_bpm, tempo_min, tempo_max]
        )

        key_mode.change(
            update_key_controls,
            inputs=[key_mode],
            outputs=[musical_key]
        )

        # Main generation function
        generate_btn.click(
            fn=process_generation,
            inputs=[
                midi_file, soundfont_file, duration, instruments, voices, style,
                tempo_mode, tempo_bpm, tempo_min, tempo_max,
                key_mode, musical_key, seed, export_formats
            ],
            outputs=[
                status_text, analysis_table, audio_output,
                download_midi, download_wav, download_mp3,
                download_analysis, download_report
            ]
        )

    return interface


def process_generation(
    midi_file,
    soundfont_file,
    duration: int,
    instruments: List[str],
    voices: int,
    style: str,
    tempo_mode: str,
    tempo_bpm: int,
    tempo_min: int,
    tempo_max: int,
    key_mode: str,
    musical_key: str,
    seed: int,
    export_formats: List[str]
) -> Tuple[Any, ...]:
    """
    Process the generation request.

    Returns:
        Tuple of outputs for Gradio interface
    """
    try:
        # Clear previous temp files
        ui_state.clear_temp_files()

        # Validate inputs
        if midi_file is None:
            return _error_response("Please upload a MIDI file")

        if not instruments:
            return _error_response("Please select at least one instrument")

        if not export_formats:
            return _error_response("Please select at least one export format")

        # Create temporary directory for processing
        temp_dir = Path(tempfile.mkdtemp())
        ui_state.add_temp_file(temp_dir)

        # Save uploaded MIDI file
        midi_path = temp_dir / "input.mid"
        with open(midi_path, "wb") as f:
            f.write(midi_file)
        ui_state.add_temp_file(midi_path)

        # Handle SoundFont file
        soundfont_path = None
        if soundfont_file is not None:
            soundfont_path = temp_dir / "soundfont.sf2"
            with open(soundfont_path, "wb") as f:
                f.write(soundfont_file)
            ui_state.add_temp_file(soundfont_path)

        # Parse parameters
        instrument_list = _parse_instruments_from_ui(instruments)
        export_format_list = _parse_export_formats_from_ui(export_formats)

        # Handle tempo parameters
        tempo_bpm_val = None
        tempo_range_val = None
        if tempo_mode == "fixed":
            tempo_bpm_val = int(tempo_bpm)
        elif tempo_mode == "range":
            tempo_range_val = (int(tempo_min), int(tempo_max))

        # Handle key parameter
        musical_key_val = None
        if key_mode == "specified":
            for key in MusicalKey:
                if key.value == musical_key:
                    musical_key_val = key
                    break

        # Create configuration
        config = MusicGenConfig(
            input_path=midi_path,
            output_dir=temp_dir,
            duration_seconds=int(duration),
            instruments=instrument_list,
            voices=int(voices),
            style=style,
            tempo_bpm=tempo_bpm_val,
            tempo_range=tempo_range_val,
            key=musical_key_val,
            seed=int(seed),
            soundfont_path=soundfont_path,
            export_formats=export_format_list
        )

        # Generate arrangement
        result = generate_arrangement(config)

        if not result.success:
            return _error_response(f"Generation failed: {result.error_message}")

        # Store result in state
        ui_state.current_result = result

        # Get generation summary
        summary = get_generation_summary(result)

        # Prepare outputs
        return _success_response(summary, result.output_files, temp_dir)

    except Exception as e:
        logger.exception("Generation failed in web UI")
        return _error_response(f"Unexpected error: {str(e)}")


def _parse_instruments_from_ui(instrument_names: List[str]) -> List[Instrument]:
    """Parse instrument names from UI."""
    instruments = []
    for name in instrument_names:
        for instrument in Instrument:
            if instrument.value == name:
                instruments.append(instrument)
                break
    return instruments


def _parse_export_formats_from_ui(format_names: List[str]) -> List[ExportFormat]:
    """Parse export format names from UI."""
    formats = []
    for name in format_names:
        try:
            formats.append(ExportFormat(name))
        except ValueError:
            pass
    return formats


def _error_response(message: str) -> Tuple[Any, ...]:
    """Create error response for UI."""
    return (
        f"❌ Error: {message}",  # status_text
        pd.DataFrame(),  # analysis_table
        None,  # audio_output
        None,  # download_midi
        None,  # download_wav
        None,  # download_mp3
        None,  # download_analysis
        None,  # download_report
    )


def _success_response(
    summary: dict,
    output_files: dict,
    temp_dir: Path
) -> Tuple[Any, ...]:
    """Create success response for UI."""
    # Create analysis table
    analysis = summary["input_analysis"]
    analysis_data = [
        ["Key", analysis["key"]],
        ["Tempo", f"{analysis['tempo']} BPM"],
        ["Time Signature", analysis["time_signature"]],
        ["Duration", f"{analysis['duration']} seconds"],
        ["Note Density", f"{analysis['note_density']} notes/sec"],
        ["Instruments", str(analysis["instruments"])]
    ]
    analysis_df = pd.DataFrame(analysis_data, columns=["Property", "Value"])

    # Prepare file downloads
    midi_file = output_files.get("midi")
    wav_file = output_files.get("wav")
    mp3_file = output_files.get("mp3")
    analysis_file = output_files.get("analysis")
    report_file = output_files.get("report")

    # For audio preview, use WAV or MP3
    audio_preview = wav_file or mp3_file

    status_message = "✅ Generation completed successfully!"

    return (
        status_message,
        analysis_df,
        str(audio_preview) if audio_preview else None,
        str(midi_file) if midi_file else None,
        str(wav_file) if wav_file else None,
        str(mp3_file) if mp3_file else None,
        str(analysis_file) if analysis_file else None,
        str(report_file) if report_file else None,
    )


def launch_web_ui(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    debug: bool = False
) -> None:
    """
    Launch the web UI.

    Args:
        host: Host address
        port: Port number
        share: Create shareable public link
        debug: Enable debug mode
    """
    config = WebUIConfig(
        host=host,
        port=port,
        share=share,
        debug=debug
    )

    interface = create_interface(config)

    logger.info(f"Launching web UI at http://{host}:{port}")

    try:
        interface.launch(
            server_name=host,
            server_port=port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=not debug
        )
    except KeyboardInterrupt:
        logger.info("Web UI stopped by user")
    except Exception as e:
        logger.error(f"Failed to launch web UI: {e}")
        raise
    finally:
        # Clean up temporary files
        ui_state.clear_temp_files()


def main() -> None:
    """Main entry point for the web UI."""
    import argparse

    parser = argparse.ArgumentParser(description="MusicGen Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create shareable link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    launch_web_ui(
        host=args.host,
        port=args.port,
        share=args.share,
        debug=args.debug
    )


if __name__ == "__main__":
    main()