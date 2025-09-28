"""
Web UI for MusicGen using Gradio.

This module provides a user-friendly web interface for:
- File upload (MIDI and SoundFont)
- Parameter configuration
- Real-time generation
- Audio preview and download
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from .config import ExportFormat, Instrument, MusicalKey, MusicGenConfig, WebUIConfig
from .orchestration import generate_arrangement, get_generation_summary

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


def create_interface(config: WebUIConfig | None = None) -> gr.Blocks:
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
        .compact-row {
            gap: 0.5rem;
        }
        """,
        head="""
        <style>
        /* Hide 404 errors in console */
        [data-testid="error"] { display: none !important; }
        </style>
        <link rel="manifest" href="data:application/json;base64,e30=" />
        """,
    ) as interface:
        gr.Markdown(
            """
            # 🎵 MusicGen
            Generate musical arrangements from MIDI files. Upload MIDI → Configure → Generate → Download
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                # File upload section
                gr.Markdown("### 📁 Files")
                with gr.Row():
                    midi_file = gr.File(
                        label="MIDI File",
                        file_count="single",
                        show_label=True,
                        scale=3
                    )
                    soundfont_file = gr.File(
                        label="SoundFont (Optional)",
                        file_types=[".sf2"],
                        file_count="single",
                        show_label=True,
                        scale=2
                    )

                # Generation parameters
                gr.Markdown("### ⚙️ Parameters")

                with gr.Row():
                    duration = gr.Slider(
                        minimum=10,
                        maximum=300,
                        value=120,
                        step=10,
                        label="Duration (sec)",
                        scale=1
                    )
                    voices = gr.Slider(
                        minimum=1, maximum=8, value=2, step=1, label="Voices", scale=1
                    )

                with gr.Row():
                    instruments = gr.CheckboxGroup(
                        choices=[inst.value for inst in Instrument],
                        value=["piano"],
                        label="Instruments",
                        scale=2
                    )
                    style = gr.Textbox(
                        value="classical",
                        label="Style",
                        placeholder="classical, jazz, rock, ambient",
                        scale=1
                    )

                # Musical parameters
                with gr.Accordion("🎼 Advanced", open=False):
                    tempo_mode = gr.Radio(
                        choices=["auto", "fixed", "range"],
                        value="auto",
                        label="Tempo Mode",
                    )

                    tempo_bpm = gr.Slider(
                        minimum=60,
                        maximum=200,
                        value=120,
                        step=1,
                        label="Fixed Tempo (BPM)",
                        visible=False,
                    )

                    with gr.Row():
                        tempo_min = gr.Slider(
                            minimum=60,
                            maximum=200,
                            value=90,
                            step=1,
                            label="Min",
                            visible=False,
                        )
                        tempo_max = gr.Slider(
                            minimum=60,
                            maximum=200,
                            value=120,
                            step=1,
                            label="Max",
                            visible=False,
                        )

                    key_mode = gr.Radio(
                        choices=["auto", "specified"], value="auto", label="Key Mode"
                    )

                    musical_key = gr.Dropdown(
                        choices=[key.value for key in MusicalKey],
                        value="C major",
                        label="Musical Key",
                        visible=False,
                    )

                    with gr.Row():
                        seed = gr.Number(value=42, label="Seed", precision=0, scale=1)
                        export_formats = gr.CheckboxGroup(
                            choices=[fmt.value for fmt in ExportFormat],
                            value=["midi", "wav", "mp3"],
                            label="Export",
                            scale=2
                        )

                # Generate button
                generate_btn = gr.Button(
                    "🎵 Generate Arrangement", variant="primary", size="lg"
                )

            with gr.Column(scale=3):
                # Results section
                gr.Markdown("### 📊 Results")

                status_text = gr.Textbox(
                    label="Status", value="Ready to generate...", interactive=False, max_lines=1
                )

                # Analysis results
                with gr.Row():
                    analysis_table = gr.DataFrame(
                        label="Analysis",
                        headers=["Property", "Value"],
                        datatype=["str", "str"],
                        elem_classes=["analysis-table"],
                        scale=2
                    )
                    # Audio preview
                    audio_output = gr.Audio(label="Preview", visible=False, scale=1)

                # Download links
                gr.Markdown("### 📥 Downloads")
                with gr.Row():
                    download_midi = gr.File(label="MIDI", visible=False, scale=1)
                    download_wav = gr.File(label="WAV", visible=False, scale=1)
                    download_mp3 = gr.File(label="MP3", visible=False, scale=1)

                with gr.Row():
                    download_analysis = gr.File(label="Analysis", visible=False, scale=1)
                    download_report = gr.File(label="Report", visible=False, scale=1)

        # Event handlers
        def update_tempo_controls(mode):
            """Update tempo control visibility based on mode."""
            return {
                tempo_bpm: gr.update(visible=(mode == "fixed")),
                tempo_min: gr.update(visible=(mode == "range")),
                tempo_max: gr.update(visible=(mode == "range")),
            }

        def update_key_controls(mode):
            """Update key control visibility based on mode."""
            return {musical_key: gr.update(visible=(mode == "specified"))}

        tempo_mode.change(
            update_tempo_controls,
            inputs=[tempo_mode],
            outputs=[tempo_bpm, tempo_min, tempo_max],
        )

        key_mode.change(update_key_controls, inputs=[key_mode], outputs=[musical_key])

        # Main generation function
        generate_btn.click(
            fn=process_generation,
            inputs=[
                midi_file,
                soundfont_file,
                duration,
                instruments,
                voices,
                style,
                tempo_mode,
                tempo_bpm,
                tempo_min,
                tempo_max,
                key_mode,
                musical_key,
                seed,
                export_formats,
            ],
            outputs=[
                status_text,
                analysis_table,
                audio_output,
                download_midi,
                download_wav,
                download_mp3,
                download_analysis,
                download_report,
            ],
            show_progress=True,
        )

    return interface


def process_generation(
    midi_file,
    soundfont_file,
    duration: int,
    instruments: list[str],
    voices: int,
    style: str,
    tempo_mode: str,
    tempo_bpm: int,
    tempo_min: int,
    tempo_max: int,
    key_mode: str,
    musical_key: str,
    seed: int,
    export_formats: list[str],
) -> tuple[Any, ...]:
    """
    Process the generation request.

    Returns:
        Tuple of outputs for Gradio interface
    """
    try:
        # Clear previous temp files
        ui_state.clear_temp_files()

        # Debug logging
        logger.debug(f"Received MIDI file: {midi_file}")
        logger.debug(f"Received soundfont file: {soundfont_file}")
        logger.debug(f"Received instruments: {instruments}")
        logger.debug(f"Received export formats: {export_formats}")

        # Validate inputs
        if midi_file is None:
            return _error_response("Please upload a MIDI file")

        if not instruments:
            return _error_response("Please select at least one instrument")

        if not export_formats:
            return _error_response("Please select at least one export format")

        # Create output directory for processing (accessible to user)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path("webui_output")
        output_base.mkdir(exist_ok=True)
        temp_dir = output_base / f"generation_{timestamp}"
        temp_dir.mkdir(exist_ok=True)
        ui_state.add_temp_file(temp_dir)

        # Save uploaded MIDI file
        midi_path = temp_dir / "input.mid"
        # In Gradio 5.x, midi_file is either a file path string or a file object
        if hasattr(midi_file, 'name'):
            # It's a file object with .name attribute
            source_path = midi_file.name
        else:
            # It's a file path string
            source_path = midi_file

        with open(midi_path, "wb") as dest_f:
            with open(source_path, "rb") as src_f:
                dest_f.write(src_f.read())
        ui_state.add_temp_file(midi_path)

        # Handle SoundFont file
        soundfont_path = None
        if soundfont_file is not None:
            soundfont_path = temp_dir / "soundfont.sf2"
            # In Gradio 5.x, soundfont_file is either a file path string or a file object
            if hasattr(soundfont_file, 'name'):
                # It's a file object with .name attribute
                source_sf_path = soundfont_file.name
            else:
                # It's a file path string
                source_sf_path = soundfont_file

            with open(soundfont_path, "wb") as dest_f:
                with open(source_sf_path, "rb") as src_f:
                    dest_f.write(src_f.read())
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
            export_formats=export_format_list,
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


def _parse_instruments_from_ui(instrument_names: list[str]) -> list[Instrument]:
    """Parse instrument names from UI."""
    instruments = []
    for name in instrument_names:
        for instrument in Instrument:
            if instrument.value == name:
                instruments.append(instrument)
                break
    return instruments


def _parse_export_formats_from_ui(format_names: list[str]) -> list[ExportFormat]:
    """Parse export format names from UI."""
    formats = []
    for name in format_names:
        try:
            formats.append(ExportFormat(name))
        except ValueError:
            pass
    return formats


def _error_response(message: str) -> tuple[Any, ...]:
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
    summary: dict, output_files: dict, temp_dir: Path
) -> tuple[Any, ...]:
    """Create success response for UI."""
    # Create analysis table
    analysis = summary["input_analysis"]
    analysis_data = [
        ["Key", analysis["key"]],
        ["Tempo", f"{analysis['tempo']} BPM"],
        ["Time Signature", analysis["time_signature"]],
        ["Duration", f"{analysis['duration']} seconds"],
        ["Note Density", f"{analysis['note_density']} notes/sec"],
        ["Instruments", str(analysis["instruments"])],
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
        str(audio_preview) if audio_preview and audio_preview.exists() else None,
        str(midi_file) if midi_file and midi_file.exists() else None,
        str(wav_file) if wav_file and wav_file.exists() else None,
        str(mp3_file) if mp3_file and mp3_file.exists() else None,
        str(analysis_file) if analysis_file and analysis_file.exists() else None,
        str(report_file) if report_file and report_file.exists() else None,
    )


def launch_web_ui(
    host: str = "127.0.0.1", port: int = 7860, share: bool = False, debug: bool = False
) -> None:
    """
    Launch the web UI.

    Args:
        host: Host address
        port: Port number
        share: Create shareable public link
        debug: Enable debug mode
    """
    config = WebUIConfig(host=host, port=port, share=share, debug=debug)

    interface = create_interface(config)

    logger.info(f"Launching web UI at http://{host}:{port}")

    try:
        interface.launch(
            server_name=host,
            server_port=port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=not debug,
            inbrowser=False,  # Don't auto-open browser
            show_api=debug,   # Show API docs in debug mode
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
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    launch_web_ui(host=args.host, port=args.port, share=args.share, debug=args.debug)


if __name__ == "__main__":
    main()
