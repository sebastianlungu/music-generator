# 🎵 MusicGen

A typed, testable Python tool for MIDI analysis and musical arrangement generation.

**MusicGen** analyzes MIDI files and generates new musical arrangements using rule-based algorithms and optional machine learning. It provides both command-line and web interfaces for easy use.

## ✨ Features

- **MIDI Analysis**: Detect key, tempo, time signature, structure, and musical features
- **Arrangement Generation**: Create new multi-voice arrangements with configurable instruments
- **Audio Synthesis**: Render arrangements to audio using SoundFonts (FluidSynth)
- **Multiple Interfaces**: Command-line tool and web UI
- **Flexible Output**: Export to MIDI, WAV, and MP3 formats
- **Reproducible**: Deterministic generation with seed control
- **Extensible**: Clean, typed codebase with comprehensive tests

## 🚀 Quick Start

### Installation

1. **Install Python 3.10+** (required)

2. **Install uv** (recommended package manager):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Clone and install MusicGen**:
   ```bash
   git clone <repository-url>
   cd musicgen
   uv sync
   ```

4. **Install system dependencies**:

   **For audio synthesis (optional but recommended):**
   - **macOS**: `brew install fluidsynth`
   - **Ubuntu/Debian**: `sudo apt-get install fluidsynth libfluidsynth-dev`
   - **Windows**: Download FluidSynth from the official website

   **For MP3 export (optional):**
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **Windows**: Download FFmpeg from the official website

### Get a SoundFont

For audio synthesis, you'll need a SoundFont (.sf2) file:

- **Free option**: [GeneralUser GS](https://schristiancollins.com/generaluser.php)
- **Higher quality**: [FluidR3_GM.sf2](https://musescore.org/en/handbook/3/soundfonts-and-sfz-files#gm_soundfonts)

Download and save it somewhere accessible (e.g., `./assets/GeneralUser.sf2`).

### Basic Usage

**Command Line:**
```bash
# Analyze a MIDI file
uv run python -m musicgen.cli analyze input.mid

# Generate arrangement
uv run python -m musicgen.cli generate input.mid \
  --output ./output \
  --duration-seconds 120 \
  --instruments "piano,guitar,strings" \
  --voices 3 \
  --style "classical" \
  --soundfont ./assets/GeneralUser.sf2
```

**Web Interface:**
```bash
# Launch web UI
uv run python -m musicgen.webui

# Then open http://localhost:7860 in your browser
```

## 📖 Documentation

### Command Line Interface

#### Generate Command

```bash
uv run python -m musicgen.cli generate [OPTIONS] INPUT_PATH
```

**Arguments:**
- `INPUT_PATH`: MIDI file or directory containing MIDI files

**Options:**
- `--output, -o`: Output directory (default: `./out`)
- `--duration-seconds, -d`: Maximum duration in seconds (default: 120)
- `--instruments, -i`: Comma-separated instrument list (default: `piano`)
- `--voices, -v`: Number of voices/parts (default: 1)
- `--style, -s`: Musical style description (default: `classical`)
- `--tempo-bpm, -t`: Fixed tempo in BPM
- `--tempo-range, -tr`: Tempo range as `min:max` (e.g., `90:120`)
- `--key, -k`: Target musical key (e.g., `C major`, `A minor`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--soundfont, -sf`: Path to SoundFont file (.sf2)
- `--export, -e`: Export formats: `midi,wav,mp3` (default: all)
- `--batch, -b`: Process all MIDI files in directory
- `--verbose, -v`: Enable verbose output

**Available Instruments:**
`piano`, `guitar`, `violin`, `cello`, `flute`, `clarinet`, `trumpet`, `saxophone`, `voice`, `choir`, `string_ensemble`, `brass_ensemble`, `woodwind_ensemble`, `organ`, `bass`, `drums`

**Available Keys:**
All major and minor keys (e.g., `C major`, `G major`, `A minor`, `F# minor`)

#### Analyze Command

```bash
uv run python -m musicgen.cli analyze [OPTIONS] INPUT_PATH
```

**Options:**
- `--output, -o`: Save analysis to JSON file
- `--verbose, -v`: Enable verbose output

#### Info Command

```bash
uv run python -m musicgen.cli info
```

Shows system capabilities and available options.

### Web Interface

Launch the web interface:

```bash
uv run python -m musicgen.webui [OPTIONS]
```

**Options:**
- `--host`: Host address (default: `127.0.0.1`)
- `--port`: Port number (default: `7860`)
- `--share`: Create public shareable link
- `--debug`: Enable debug mode

The web interface provides:
- File upload for MIDI and SoundFont files
- Interactive parameter controls
- Real-time audio preview
- Download links for all generated files

### Python API

```python
from pathlib import Path
from musicgen import MusicGenConfig, generate_arrangement
from musicgen.config import Instrument, ExportFormat

# Create configuration
config = MusicGenConfig(
    input_path=Path("input.mid"),
    output_dir=Path("./output"),
    duration_seconds=120,
    instruments=[Instrument.PIANO, Instrument.GUITAR],
    voices=2,
    style="jazz",
    soundfont_path=Path("./assets/GeneralUser.sf2"),
    export_formats=[ExportFormat.MIDI, ExportFormat.WAV],
    seed=42
)

# Generate arrangement
result = generate_arrangement(config)

if result.success:
    print(f"Generated files: {result.output_files}")
else:
    print(f"Generation failed: {result.error_message}")
```

## 📁 Output Structure

For each generation, MusicGen creates a directory with:

```
output/
└── input_filename/
    ├── analysis.json      # Input analysis results
    ├── generated.mid      # Generated MIDI file
    ├── render.wav         # Rendered audio (if SoundFont provided)
    ├── render.mp3         # MP3 version (if requested)
    └── report.txt         # Generation parameters and rationale
```

### Analysis JSON Format

```json
{
  "key": "C major",
  "tempo": 120.0,
  "time_signature": [4, 4],
  "duration_seconds": 45.2,
  "pitch_histogram": [0.15, 0.02, 0.18, ...],
  "note_density": 2.3,
  "sections": [[0.0, 16.0], [16.0, 32.0], ...],
  "instrument_programs": [0, 24, 40]
}
```

## 🎼 Musical Styles

MusicGen supports various musical styles that influence harmonic progressions and rhythmic patterns:

- **`classical`**: Traditional voice leading, I-IV-V-I progressions
- **`jazz`**: Extended harmony, ii-V-I progressions, syncopation
- **`rock`**: Power chords, drum patterns, rhythmic emphasis
- **`ambient`**: Sustained chords, sparse textures, atmospheric sounds
- **`pop`**: I-V-vi-IV progressions, contemporary rhythms
- **`baroque`**: Contrapuntal textures, ornamental figures
- **`modal`**: Modal harmony and scales

You can also use free-form descriptions like `"ambient electronic"` or `"folk ballad"`.

## ⚙️ Configuration

### Environment Variables

- `MUSICGEN_DEFAULT_SOUNDFONT`: Default SoundFont path
- `MUSICGEN_TEMP_DIR`: Temporary directory for processing
- `MUSICGEN_LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

### Advanced Configuration

Create custom arrangement configurations:

```python
from musicgen.config import ArrangementConfig

arrangement_config = ArrangementConfig(
    humanization_amount=0.15,      # Add timing/velocity variation
    voice_leading_strictness=0.8,  # Enforce voice leading rules
    style_rules={
        "custom_style": {
            "harmonic_rhythm": "fast",
            "texture": "homophonic",
            "voice_leading": True
        }
    }
)

result = generate_arrangement(config, arrangement_config=arrangement_config)
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_analysis.py

# Run with coverage
uv run pytest --cov=musicgen
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix

# Type check
uv run pyright
```

### Project Structure

```
musicgen/
├── __init__.py           # Package entry point
├── cli.py               # Command-line interface
├── config.py            # Configuration models
├── io_files.py          # File I/O operations
├── analysis.py          # MIDI analysis functions
├── arrange.py           # Arrangement generation
├── orchestration.py     # High-level pipeline coordination
├── synthesis.py         # Audio synthesis
└── webui.py            # Web interface

tests/
├── test_analysis.py     # Analysis function tests
├── test_arrange.py      # Arrangement generation tests
├── test_cli_smoke.py    # CLI integration tests
└── test_webui_smoke.py  # Web UI tests
```

## 🚫 Limitations

- **Audio synthesis requires FluidSynth**: Without it, only MIDI export is available
- **MP3 export requires FFmpeg**: WAV export works without it
- **No real-time performance**: Designed for offline generation
- **Limited to General MIDI**: Instrument selection based on GM standard
- **No advanced mixing**: Simple audio rendering without effects or mastering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Ensure tests pass: `uv run pytest`
5. Check code quality: `uv run ruff format . && uv run ruff check . --fix`
6. Submit a pull request

### Code Standards

- Python 3.10+ with type hints
- Functions ≤50 lines with docstrings
- Files ≤500 lines
- Pure functions where possible
- Comprehensive tests for new features

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pretty MIDI**: MIDI file processing
- **Music21**: Musical analysis algorithms
- **FluidSynth**: High-quality audio synthesis
- **Gradio**: Web interface framework
- **Typer**: Command-line interface
- **Pydantic**: Configuration validation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/musicgen/musicgen/issues)
- **Discussions**: [GitHub Discussions](https://github.com/musicgen/musicgen/discussions)
- **Documentation**: [Wiki](https://github.com/musicgen/musicgen/wiki)

---

**Made with ❤️ for musicians and developers**