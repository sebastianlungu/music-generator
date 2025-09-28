# 🎵 MusicGen - Quick Start Guide

## ✅ Installation Complete!

Your MusicGen virtual environment is set up and ready to use!

## 🚀 How to Run

### 1. Activate Environment (Required for each new terminal)

```bash
# Method 1: Using source (Git Bash/WSL)
source .venv/Scripts/activate

# Method 2: Using uv (Windows)
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Verify Installation

```bash
python -c "print('MusicGen is ready!')"
```

## 🎼 Usage Examples

### CLI Commands

```bash
# Show help
python -m musicgen.cli --help

# Show system info (will warn about missing FluidSynth - that's normal)
python -c "from musicgen import synthesis; print('Audio available:', synthesis.check_fluidsynth_available())"

# For MIDI analysis and generation, you'll need a MIDI file
# Download one or create one with your DAW

# Analyze a MIDI file
python -m musicgen.cli analyze your_file.mid

# Generate arrangement (MIDI output only)
python -m musicgen.cli generate your_file.mid \
  --output ./output \
  --duration-seconds 60 \
  --instruments "piano,guitar" \
  --voices 2 \
  --export midi
```

### Web Interface

```bash
# Launch web UI
python -m musicgen.webui

# Then open http://localhost:7860 in your browser
```

### Python API

```python
from pathlib import Path
from musicgen.config import MusicGenConfig, Instrument, ExportFormat
from musicgen.orchestration import generate_arrangement

config = MusicGenConfig(
    input_path=Path("your_file.mid"),
    output_dir=Path("./output"),
    duration_seconds=60,
    instruments=[Instrument.PIANO, Instrument.GUITAR],
    voices=2,
    style="classical",
    export_formats=[ExportFormat.MIDI]
)

result = generate_arrangement(config)
```

## 📁 What You Have

```
BGM/
├── .venv/              # Virtual environment (activated with source .venv/Scripts/activate)
├── musicgen/           # Main package
├── tests/              # Test suite
├── pyproject.toml      # Package configuration
├── README.md           # Full documentation
├── claude.md           # Development guide
├── SETUP.md            # Detailed setup instructions
└── QUICK_START.md      # This file
```

## 🔧 Current Status

✅ **Working:**
- Core MIDI processing and analysis
- Musical arrangement generation
- CLI and Web interfaces
- Complete test suite
- All Python dependencies installed

⚠️ **Limited (Windows-specific):**
- Audio synthesis (needs FluidSynth system install)
- MP3 export (needs FFmpeg system install)

## 💡 Next Steps

1. **Find a MIDI file** to experiment with
2. **Try the CLI commands** above
3. **Launch the web interface** for easier interaction
4. **Check the full README.md** for complete documentation

## 🛠️ For Audio Support (Optional)

If you want WAV/MP3 output, install system dependencies:

```bash
# Windows (using Chocolatey)
choco install fluidsynth ffmpeg

# Or download manually from:
# FluidSynth: https://github.com/FluidSynth/fluidsynth/releases
# FFmpeg: https://ffmpeg.org/download.html
```

## 🎵 Example Workflow

1. Get a MIDI file (download from internet or create in DAW)
2. Analyze: `python -m musicgen.cli analyze input.mid`
3. Generate: `python -m musicgen.cli generate input.mid --export midi`
4. Check `./out/` directory for results
5. Open generated MIDI in your music software

---

**Your MusicGen installation is ready! Start creating music! 🎼**

Need help? Check `README.md` for complete documentation.