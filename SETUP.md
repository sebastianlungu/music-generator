# MusicGen Setup Instructions

## ✅ Current Status

Your MusicGen installation is **working** with the following capabilities:

- ✅ **Core MIDI processing** - Ready to use
- ✅ **Analysis algorithms** - Key, tempo, structure detection
- ✅ **Arrangement generation** - Rule-based composition
- ✅ **CLI interface** - Command-line tools
- ✅ **Web interface** - Gradio-based web UI
- ⚠️ **Audio synthesis** - Disabled (no FluidSynth)
- ⚠️ **MP3 export** - Disabled (no FFmpeg)

## 🚀 How to Run

### 1. Activate the Environment

```bash
# Windows (Git Bash/WSL)
source .venv/Scripts/activate

# Or use uv directly (Windows)
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Test the Installation

```bash
# Quick test
python test_simple.py

# CLI help
python -m musicgen.cli --help
```

### 3. Basic Usage Examples

#### Analyze a MIDI file (MIDI-only mode):
```bash
# You'll need a MIDI file to test with
python -m musicgen.cli analyze path/to/your/file.mid
```

#### Generate arrangement (MIDI-only):
```bash
python -m musicgen.cli generate path/to/your/file.mid \
  --output ./output \
  --duration-seconds 60 \
  --instruments "piano,guitar" \
  --voices 2 \
  --style "classical" \
  --export midi
```

#### Launch Web Interface:
```bash
python -m musicgen.webui
# Then open http://localhost:7860 in your browser
```

## 🎵 What Works Right Now

**MIDI Generation:** The system can:
- Analyze existing MIDI files for key, tempo, structure
- Generate new musical arrangements based on analysis
- Create multi-voice compositions with different instruments
- Export to MIDI format
- Support various musical styles (classical, jazz, rock, ambient)

**Interfaces:** Both CLI and web interfaces work for MIDI-only workflow.

## 🔧 Optional: Enable Audio Synthesis

If you want audio output (WAV/MP3), you'll need to install system dependencies:

### Windows (Advanced Users)

1. **Install FluidSynth:**
   - Download from: https://github.com/FluidSynth/fluidsynth/releases
   - Or use Chocolatey: `choco install fluidsynth`

2. **Install FFmpeg:**
   - Download from: https://ffmpeg.org/download.html
   - Or use Chocolatey: `choco install ffmpeg`

3. **Get a SoundFont:**
   - Download [GeneralUser GS](https://schristiancollins.com/generaluser.php)
   - Save as `./assets/GeneralUser.sf2`

### Alternative: Use Docker (Easier)

```bash
# Build Docker image with all dependencies
docker build -t musicgen .

# Run with audio support
docker run -it -p 7860:7860 musicgen
```

## 📁 Project Structure

```
musicgen/
├── __init__.py          # Main package
├── cli.py              # Command-line interface
├── config.py           # Configuration models
├── analysis.py         # MIDI analysis
├── arrange.py          # Music generation
├── orchestration.py    # Pipeline coordination
├── synthesis.py        # Audio rendering (needs FluidSynth)
├── webui.py           # Web interface
└── io_files.py        # File I/O operations

tests/                  # Comprehensive test suite
pyproject.toml         # Package configuration
README.md              # Full documentation
```

## 🛠️ Development Commands

```bash
# Run tests
python -m pytest

# Format code
python -m ruff format .

# Type check
python -m pyright

# Lint
python -m ruff check . --fix
```

## 🎼 Example Workflow

1. **Find a MIDI file** (or download one from the internet)
2. **Analyze it:** `python -m musicgen.cli analyze input.mid`
3. **Generate arrangement:** `python -m musicgen.cli generate input.mid --export midi`
4. **Check output** in `./out/` directory
5. **Open generated MIDI** in your favorite MIDI player/DAW

## 🚫 Current Limitations

- **No audio output** without FluidSynth installation
- **No MP3 export** without FFmpeg installation
- **Windows console encoding** issues with some Unicode characters
- **Analysis depends on music21** which may have slow first-time startup

## 💡 Tips

- **Start with MIDI-only** workflow to test the system
- **Use relative paths** for input files to avoid path issues
- **Check the `./out/` directory** for generated files
- **Try different styles:** classical, jazz, rock, ambient, pop
- **Experiment with instruments:** piano, guitar, violin, drums, etc.

---

**Your MusicGen installation is ready to create music! 🎵**