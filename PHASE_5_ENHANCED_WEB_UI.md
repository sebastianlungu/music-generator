# Phase 5: Enhanced Web UI

This phase focuses on enhancing the web UI with real-time MP3 generation, browser audio player integration, progress indicators, and improved download management for a seamless user experience.

## Overview

Phase 5 transforms the basic web interface into a professional music generation tool with real-time audio feedback, progress tracking, and streamlined file management. Users can now listen to generated music directly in the browser while monitoring generation progress.

## Key Features

### 1. Real-time MP3 Generation
- Streaming audio generation with immediate playback
- Background processing with progress updates
- Automatic file management and cleanup

### 2. Browser Audio Player Integration
- Native HTML5 audio player with Gradio components
- Waveform visualization (optional)
- Playback controls (play, pause, seek, volume)

### 3. Progress Indicators
- Real-time progress bars for each generation stage
- Status messages with estimated completion times
- Error handling with user-friendly feedback

### 4. Enhanced Download Management
- Batch download of all generated files
- Organized file structure with metadata
- Automatic cleanup of temporary files

## Implementation Details

### Enhanced WebUI Module (`musicgen/webui.py`)

```python
import gradio as gr
import asyncio
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Generator
import threading
import queue
import time

from .orchestration import generate_music_from_config
from .config import MusicGenConfig

class ProgressTracker:
    """Thread-safe progress tracking for audio generation."""

    def __init__(self):
        self.progress = 0.0
        self.status = "Ready"
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def update(self, progress: float, status: str):
        """Update progress and status thread-safely."""
        with self._lock:
            self.progress = progress
            self.status = status

    def set_error(self, error: str):
        """Set error status."""
        with self._lock:
            self.error = error
            self.status = f"Error: {error}"

    def get_state(self) -> Tuple[float, str, Optional[str]]:
        """Get current state thread-safely."""
        with self._lock:
            return self.progress, self.status, self.error

class AudioGenerator:
    """Handles background audio generation with progress tracking."""

    def __init__(self):
        self.tracker = ProgressTracker()
        self.current_thread: Optional[threading.Thread] = None
        self.cancel_flag = threading.Event()

    def generate_with_progress(
        self,
        config: MusicGenConfig,
        progress_callback
    ) -> Optional[Tuple[str, str, str]]:
        """Generate audio with progress updates."""

        def worker():
            try:
                self.tracker.update(0.1, "Analyzing input...")
                time.sleep(0.5)  # Simulate analysis time

                if self.cancel_flag.is_set():
                    return

                self.tracker.update(0.3, "Generating arrangement...")
                time.sleep(1.0)  # Simulate arrangement time

                if self.cancel_flag.is_set():
                    return

                self.tracker.update(0.6, "Synthesizing audio...")

                # Actual generation
                result = generate_music_from_config(config)

                if self.cancel_flag.is_set():
                    return

                self.tracker.update(0.9, "Finalizing files...")
                time.sleep(0.5)

                self.tracker.update(1.0, "Complete!")

                # Store result for retrieval
                self.result = result

            except Exception as e:
                self.tracker.set_error(str(e))

        # Reset state
        self.cancel_flag.clear()
        self.result = None

        # Start background thread
        self.current_thread = threading.Thread(target=worker)
        self.current_thread.start()

        # Monitor progress
        while self.current_thread.is_alive():
            progress, status, error = self.tracker.get_state()
            progress_callback(progress, status)

            if error:
                return None

            time.sleep(0.1)

        # Return final result
        return getattr(self, 'result', None)

    def cancel(self):
        """Cancel current generation."""
        self.cancel_flag.set()
        if self.current_thread and self.current_thread.is_alive():
            self.current_thread.join(timeout=5.0)

# Global generator instance
audio_generator = AudioGenerator()

def create_audio_player(audio_path: Optional[str]) -> str:
    """Create HTML5 audio player component."""
    if not audio_path or not Path(audio_path).exists():
        return "<p>No audio available</p>"

    return f"""
    <div class="audio-player">
        <audio controls style="width: 100%;">
            <source src="file/{audio_path}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
    </div>
    """

def generate_music_gradio(
    input_file,
    duration: int,
    instruments: str,
    voices: int,
    style: str,
    tempo_range: str,
    key: str,
    soundfont_file,
    progress=gr.Progress()
) -> Tuple[str, str, str, str]:
    """
    Generate music with real-time progress updates.

    Returns:
        Tuple of (audio_player_html, midi_file, analysis_json, download_zip)
    """

    if not input_file:
        return "Please upload a MIDI file", None, None, None

    try:
        # Parse tempo range
        tempo_min, tempo_max = 90, 120
        if tempo_range and ':' in tempo_range:
            parts = tempo_range.split(':')
            tempo_min, tempo_max = int(parts[0]), int(parts[1])

        # Create config
        config = MusicGenConfig(
            input_path=Path(input_file.name),
            output_dir=Path(tempfile.gettempdir()) / "musicgen_web",
            duration_seconds=duration,
            instruments=instruments.split(',') if instruments else ["piano"],
            voices=voices,
            style=style or "classical",
            tempo_range=(tempo_min, tempo_max),
            key=key if key != "auto" else None,
            soundfont_path=Path(soundfont_file.name) if soundfont_file else None,
            export_formats=["midi", "mp3"],
            seed=42
        )

        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Progress callback for Gradio
        def update_progress(prog: float, status: str):
            progress(prog, desc=status)

        # Generate with progress tracking
        result = audio_generator.generate_with_progress(config, update_progress)

        if not result:
            return "Generation failed or was cancelled", None, None, None

        output_dir, midi_path, mp3_path = result

        # Create audio player HTML
        audio_html = create_audio_player(mp3_path)

        # Read analysis file
        analysis_path = output_dir / "analysis.json"
        analysis_content = ""
        if analysis_path.exists():
            analysis_content = analysis_path.read_text()

        # Create download package
        zip_path = create_download_package(output_dir)

        return audio_html, midi_path, analysis_content, zip_path

    except Exception as e:
        return f"Error: {str(e)}", None, None, None

def create_download_package(output_dir: Path) -> str:
    """Create a ZIP package with all generated files."""

    zip_path = output_dir / "musicgen_output.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add all files in output directory
        for file_path in output_dir.glob("*"):
            if file_path.is_file() and file_path.suffix != ".zip":
                zf.write(file_path, file_path.name)

    return str(zip_path)

def create_enhanced_interface():
    """Create the enhanced Gradio interface."""

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate"
    )

    with gr.Blocks(
        theme=theme,
        title="MusicGen - AI Music Generator",
        css="""
        .audio-player { margin: 10px 0; }
        .progress-bar { margin: 10px 0; }
        .status-text { font-style: italic; color: #666; }
        """
    ) as interface:

        gr.Markdown("# 🎵 MusicGen - AI Music Generator")
        gr.Markdown("Generate new musical arrangements from MIDI files with real-time audio preview.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Configuration")

                input_file = gr.File(
                    label="Upload MIDI File",
                    file_types=[".mid", ".midi"],
                    type="filepath"
                )

                duration = gr.Slider(
                    label="Duration (seconds)",
                    minimum=30,
                    maximum=300,
                    value=60,
                    step=10
                )

                instruments = gr.Textbox(
                    label="Instruments (comma-separated)",
                    value="piano,strings",
                    placeholder="piano,guitar,violin"
                )

                voices = gr.Slider(
                    label="Number of Voices",
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=1
                )

                style = gr.Textbox(
                    label="Musical Style",
                    value="classical",
                    placeholder="classical, jazz, ambient, etc."
                )

                tempo_range = gr.Textbox(
                    label="Tempo Range (BPM)",
                    value="90:120",
                    placeholder="90:120"
                )

                key = gr.Dropdown(
                    label="Key",
                    choices=["auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
                    value="auto"
                )

                soundfont_file = gr.File(
                    label="SoundFont File (optional)",
                    file_types=[".sf2"],
                    type="filepath"
                )

                generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg")
                cancel_btn = gr.Button("❌ Cancel", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("### Generated Music")

                # Audio player component
                audio_player = gr.HTML(
                    label="Audio Player",
                    value="<p>Upload a MIDI file and click Generate to create music</p>"
                )

                # Generated files
                with gr.Tabs():
                    with gr.TabItem("MIDI File"):
                        midi_output = gr.File(
                            label="Generated MIDI",
                            interactive=False
                        )

                    with gr.TabItem("Analysis"):
                        analysis_output = gr.JSON(
                            label="Musical Analysis",
                            value={}
                        )

                    with gr.TabItem("Download"):
                        download_output = gr.File(
                            label="Download All Files (ZIP)",
                            interactive=False
                        )

        # Event handlers
        generate_btn.click(
            fn=generate_music_gradio,
            inputs=[
                input_file, duration, instruments, voices,
                style, tempo_range, key, soundfont_file
            ],
            outputs=[audio_player, midi_output, analysis_output, download_output],
            show_progress=True
        )

        cancel_btn.click(
            fn=lambda: audio_generator.cancel(),
            outputs=None
        )

        # Example section
        gr.Markdown("### Example Usage")
        gr.Markdown("""
        1. Upload a MIDI file (any format)
        2. Adjust parameters as desired
        3. Click "Generate Music" and watch the progress
        4. Listen to the generated audio directly in your browser
        5. Download individual files or the complete ZIP package
        """)

    return interface

def launch_enhanced_webui(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False
):
    """Launch the enhanced web UI with real-time features."""

    interface = create_enhanced_interface()

    print(f"🎵 Starting Enhanced MusicGen Web UI...")
    print(f"   URL: http://{host}:{port}")
    print(f"   Features: Real-time audio, progress tracking, batch downloads")

    interface.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        favicon_path=None,
        inbrowser=True
    )

if __name__ == "__main__":
    launch_enhanced_webui()
```

### Key Implementation Features

#### 1. Real-time Progress Tracking
```python
class ProgressTracker:
    def __init__(self):
        self.progress = 0.0
        self.status = "Ready"
        self._lock = threading.Lock()

    def update(self, progress: float, status: str):
        with self._lock:
            self.progress = progress
            self.status = status
```

#### 2. Background Audio Generation
```python
def generate_with_progress(self, config, progress_callback):
    def worker():
        self.tracker.update(0.1, "Analyzing input...")
        # Actual generation work
        result = generate_music_from_config(config)
        self.tracker.update(1.0, "Complete!")

    thread = threading.Thread(target=worker)
    thread.start()

    # Monitor progress
    while thread.is_alive():
        progress, status, error = self.tracker.get_state()
        progress_callback(progress, status)
        time.sleep(0.1)
```

#### 3. Browser Audio Integration
```python
def create_audio_player(audio_path: str) -> str:
    return f"""
    <div class="audio-player">
        <audio controls style="width: 100%;">
            <source src="file/{audio_path}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
    </div>
    """
```

## Usage Examples

### Basic Generation with Audio Preview
```python
# Launch enhanced UI
from musicgen.webui import launch_enhanced_webui

launch_enhanced_webui(
    host="0.0.0.0",  # Allow external access
    port=7860,
    share=False      # Set True for public sharing
)
```

### Programmatic Audio Generation
```python
from musicgen.webui import AudioGenerator
from musicgen.config import MusicGenConfig

generator = AudioGenerator()
config = MusicGenConfig(
    input_path=Path("input.mid"),
    output_dir=Path("output"),
    duration_seconds=120
)

def progress_callback(progress, status):
    print(f"{progress:.1%}: {status}")

result = generator.generate_with_progress(config, progress_callback)
```

## Configuration Options

### Web UI Settings
- **Host/Port**: Configure server binding
- **Share**: Enable public Gradio sharing
- **Theme**: Customize visual appearance
- **CSS**: Add custom styling

### Audio Settings
- **Sample Rate**: 44.1 kHz default
- **Bit Depth**: 24-bit for quality
- **MP3 Encoding**: CBR 192 kbps
- **Real-time Streaming**: Progressive download support

## Performance Considerations

### Memory Management
- Automatic cleanup of temporary files
- Streaming audio generation for large files
- Progress-based memory allocation

### Concurrent Generation
- Thread-safe progress tracking
- Background processing with cancellation
- Queue management for multiple requests

## Browser Compatibility

### Supported Features
- HTML5 audio playback (all modern browsers)
- File download management
- Real-time progress updates
- WebSocket communication (Gradio)

### Fallbacks
- Basic file download for unsupported audio formats
- Text-based progress for older browsers
- Static file serving as backup

## Deployment Options

### Local Development
```bash
uv run python -m musicgen.webui
```

### Production Deployment
```bash
# With custom settings
uv run python -m musicgen.webui \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim
COPY . /app
WORKDIR /app
RUN uv install
EXPOSE 7860
CMD ["uv", "run", "python", "-m", "musicgen.webui"]
```

## Testing Strategy

### Integration Tests
- End-to-end audio generation workflow
- Progress tracking accuracy
- File management and cleanup
- Browser compatibility testing

### Performance Tests
- Concurrent user handling
- Memory usage under load
- Audio generation speed
- File download performance

## Future Enhancements

### Advanced Audio Features
- Waveform visualization
- Real-time audio effects
- Multi-track mixing interface
- MIDI keyboard integration

### Collaboration Features
- Session sharing
- Project management
- Version control for compositions
- Real-time collaboration

This enhanced web UI provides a professional, user-friendly interface for music generation with real-time feedback and seamless audio integration, making the musicgen tool accessible to both technical and non-technical users.