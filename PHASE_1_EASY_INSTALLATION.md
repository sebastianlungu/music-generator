# Phase 1: Easy Installation Foundation

**Goal**: Make `uv sync` work out-of-the-box for anyone pulling the MusicGen repository, with graceful fallbacks for audio synthesis when system dependencies are missing.

## Overview and Goals

### Current State Problems
- **Hard FluidSynth Dependency**: `pyfluidsynth>=1.3.0` in `pyproject.toml` requires system FluidSynth installation
- **Installation Friction**: New users face complex setup with system packages, environment variables, and binary dependencies
- **Platform-Specific Issues**: Windows users need manual FluidSynth installation and PATH configuration
- **CI/CD Complexity**: Automated testing requires system package installation across multiple platforms

### Solution Strategy
Move from **hard dependencies** to **soft dependencies** with **graceful fallbacks**:

1. **Remove hard audio dependencies** from core requirements
2. **Add optional dependency groups** for different audio capabilities
3. **Implement robust fallback mechanisms** when audio synthesis unavailable
4. **Maintain full functionality** when dependencies are present
5. **Provide clear installation paths** for users who want audio features

### Success Criteria
- ✅ `uv sync` works immediately on any platform without system packages
- ✅ Core MIDI analysis and arrangement functionality always available
- ✅ Tests pass in CI without system audio dependencies
- ✅ Clear upgrade path to full audio synthesis capabilities
- ✅ Informative warnings guide users to optional installations

---

## Current Dependency Analysis

### Hard Dependencies (Always Required)
```toml
dependencies = [
    "pydantic>=2.0.0",      # Config validation - Keep
    "typer>=0.9.0",         # CLI interface - Keep
    "pretty-midi>=0.2.10",  # MIDI I/O - Keep
    "mido>=1.3.0",          # Alternative MIDI - Keep
    "music21>=9.1.0",       # Music analysis - Keep
    "numpy>=1.24.0",        # Numerical ops - Keep
    "pyfluidsynth>=1.3.0",  # ❌ REMOVE (system dependency)
    "pydub>=0.25.1",        # ❌ MOVE TO OPTIONAL (needs ffmpeg)
    "gradio>=4.0.0",        # Web UI - Keep or make optional
    "markovify>=0.9.4",     # Text generation - Keep
]
```

### Problematic Dependencies
- **`pyfluidsynth`**: Requires system FluidSynth installation
- **`pydub`**: Requires FFmpeg for MP3 conversion
- **`gradio`**: Heavy dependency, consider making optional

---

## Proposed Dependency Changes

### New `pyproject.toml` Structure

```toml
[project]
dependencies = [
    # Core dependencies (always work)
    "pydantic>=2.0.0",
    "typer>=0.9.0",
    "pretty-midi>=0.2.10",
    "mido>=1.3.0",
    "music21>=9.1.0",
    "numpy>=1.24.0",
    "markovify>=0.9.4",
]

[project.optional-dependencies]
# Development tools
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "pyright>=1.1.0",
]

# Audio synthesis capabilities
audio-synthesis = [
    "pyfluidsynth>=1.3.0",
]

# Audio format conversion
audio-conversion = [
    "pydub>=0.25.1",
]

# Full audio pipeline (synthesis + conversion)
audio-full = [
    "pyfluidsynth>=1.3.0",
    "pydub>=0.25.1",
]

# Web UI features
webui = [
    "gradio>=4.0.0",
]

# Everything (for power users)
all = [
    "pyfluidsynth>=1.3.0",
    "pydub>=0.25.1",
    "gradio>=4.0.0",
]
```

### Installation Options

```bash
# Minimal installation (MIDI only)
uv sync

# With audio synthesis
uv sync --extra audio-synthesis

# With audio conversion
uv sync --extra audio-conversion

# Full audio pipeline
uv sync --extra audio-full

# Everything including web UI
uv sync --extra all

# Development setup
uv sync --extra dev --extra all
```

---

## Code Architecture Changes

### Enhanced Capability Detection

**Current**: `audio_types.py` already has good foundation
**Enhancement**: Expand capability detection and graceful fallbacks

```python
# Enhanced audio_types.py
class AudioCapability(str, Enum):
    MIDI_ONLY = "midi_only"           # Always available
    AUDIO_SYNTHESIS = "audio_synthesis"  # FluidSynth available
    AUDIO_CONVERSION = "audio_conversion" # FFmpeg available
    FULL_AUDIO = "full_audio"         # Both available
    WEB_UI = "web_ui"                 # Gradio available

def get_installation_mode() -> AudioCapability:
    """Detect current installation mode."""
    capabilities = get_audio_capabilities()

    if AudioCapability.FULL_AUDIO in capabilities:
        return AudioCapability.FULL_AUDIO
    elif AudioCapability.AUDIO_SYNTHESIS in capabilities:
        return AudioCapability.AUDIO_SYNTHESIS
    elif AudioCapability.AUDIO_CONVERSION in capabilities:
        return AudioCapability.AUDIO_CONVERSION
    else:
        return AudioCapability.MIDI_ONLY
```

### Graceful Fallback Implementation

**Update `synthesis.py`** to handle missing dependencies elegantly:

```python
# Enhanced synthesis.py

# Safe import with informative fallbacks
try:
    import fluidsynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError as e:
    FLUIDSYNTH_AVAILABLE = False
    _log_missing_dependency("fluidsynth", "audio synthesis", e)

try:
    import pydub
    PYDUB_AVAILABLE = True
except ImportError as e:
    PYDUB_AVAILABLE = False
    _log_missing_dependency("pydub", "audio conversion", e)

def synthesize_midi(
    midi_data: "pretty_midi.PrettyMIDI",
    config: MusicGenConfig
) -> np.ndarray | None:
    """
    Synthesize MIDI to audio with graceful fallbacks.

    Returns:
        Audio data if synthesis possible, None otherwise
    """
    if not FLUIDSYNTH_AVAILABLE:
        _warn_capability_unavailable(
            AudioCapability.AUDIO_SYNTHESIS,
            "Skipping audio synthesis. Install with: uv sync --extra audio-synthesis"
        )
        return None

    if config.soundfont_path is None:
        _warn_missing_soundfont()
        return None

    # Proceed with synthesis...
    return render_midi_to_audio(midi_data, config.soundfont_path)

def _warn_capability_unavailable(capability: AudioCapability, install_hint: str):
    """Issue user-friendly warning with installation instructions."""
    warnings.warn(
        f"Audio capability '{capability.value}' not available.\n"
        f"{install_hint}\n"
        f"Continuing with MIDI-only output.",
        UserWarning,
        stacklevel=3
    )

def _warn_missing_soundfont():
    """Warn about missing SoundFont with helpful suggestions."""
    warnings.warn(
        "No SoundFont file specified. Audio synthesis requires a .sf2 file.\n"
        "Download free SoundFonts from:\n"
        "- https://sites.google.com/site/soundfonts4u/\n"
        "- https://freepats.zenvoid.org/\n"
        "Then use --soundfont path/to/file.sf2",
        UserWarning
    )
```

### CLI Export Format Handling

**Update `cli.py`** to handle export format mismatches gracefully:

```python
def validate_export_formats(
    formats: list[str],
    available_capabilities: list[AudioCapability]
) -> tuple[list[str], list[str]]:
    """
    Validate export formats against available capabilities.

    Returns:
        (supported_formats, unsupported_formats)
    """
    supported = []
    unsupported = []

    for fmt in formats:
        if fmt == "midi":
            supported.append(fmt)  # Always available
        elif fmt in ["wav", "flac"] and AudioCapability.AUDIO_SYNTHESIS in available_capabilities:
            supported.append(fmt)
        elif fmt in ["mp3", "ogg"] and AudioCapability.FULL_AUDIO in available_capabilities:
            supported.append(fmt)
        else:
            unsupported.append(fmt)

    return supported, unsupported

def handle_export_format_validation(config: MusicGenConfig) -> MusicGenConfig:
    """Update config based on available capabilities."""
    available = get_audio_capabilities()
    supported, unsupported = validate_export_formats(config.export_formats, available)

    if unsupported:
        install_hints = []
        if any(fmt in ["wav", "flac"] for fmt in unsupported):
            install_hints.append("uv sync --extra audio-synthesis")
        if any(fmt in ["mp3", "ogg"] for fmt in unsupported):
            install_hints.append("uv sync --extra audio-full")

        warnings.warn(
            f"Export formats not available: {', '.join(unsupported)}\n"
            f"Install with: {' or '.join(install_hints)}\n"
            f"Continuing with: {', '.join(supported)}"
        )

    # Update config with only supported formats
    updated_config = config.model_copy()
    updated_config.export_formats = supported or ["midi"]  # Fallback to MIDI
    return updated_config
```

---

## Testing Strategy Changes

### Core Test Requirements

**Tests MUST pass** without any system audio dependencies:

```python
# Enhanced test_cli_smoke.py

class TestCLIMinimalInstallation:
    """Test CLI works with minimal installation (no audio deps)."""

    def test_midi_only_installation(self, temp_workspace):
        """Test complete workflow with MIDI-only exports."""
        test_args = [
            "musicgen", "generate",
            str(temp_workspace["test_midi"]),
            "--output", str(temp_workspace["output_dir"]),
            "--export", "midi",  # Only MIDI export
            "--duration-seconds", "10",
        ]

        with patch("sys.argv", test_args):
            # Should always succeed regardless of audio dependencies
            main()

        # Verify MIDI and analysis outputs
        assert len(list(temp_workspace["output_dir"].glob("**/*.mid"))) > 0
        assert len(list(temp_workspace["output_dir"].glob("**/*.json"))) > 0

    def test_audio_export_fallback(self, temp_workspace):
        """Test graceful fallback when audio formats requested but unavailable."""
        test_args = [
            "musicgen", "generate",
            str(temp_workspace["test_midi"]),
            "--output", str(temp_workspace["output_dir"]),
            "--export", "midi,wav,mp3",  # Request all formats
        ]

        with patch("sys.argv", test_args), \
             patch("musicgen.synthesis.FLUIDSYNTH_AVAILABLE", False):

            # Should succeed with warnings, falling back to MIDI only
            with warnings.catch_warnings(record=True) as w:
                main()

            # Should have warned about unavailable formats
            assert any("not available" in str(warning.message) for warning in w)

        # Should still produce MIDI output
        assert len(list(temp_workspace["output_dir"].glob("**/*.mid"))) > 0

class TestAudioCapabilityDetection:
    """Test audio capability detection and fallbacks."""

    def test_capability_detection_no_deps(self):
        """Test capability detection with no audio dependencies."""
        with patch("musicgen.audio_types.fluidsynth", None), \
             patch("musicgen.audio_types.pydub", None):

            capabilities = get_audio_capabilities()
            assert AudioCapability.MIDI_ONLY in capabilities
            assert AudioCapability.AUDIO_SYNTHESIS not in capabilities

    def test_installation_instructions(self):
        """Test installation instruction generation."""
        instructions = create_installation_instructions(AudioCapability.AUDIO_SYNTHESIS)
        assert "uv sync --extra audio-synthesis" in instructions

        instructions = create_installation_instructions(AudioCapability.FULL_AUDIO)
        assert "uv sync --extra audio-full" in instructions

# Conditional audio tests (only run if dependencies available)
@pytest.mark.skipif(not FLUIDSYNTH_AVAILABLE, reason="FluidSynth not available")
class TestFullAudioPipeline:
    """Test full audio pipeline when dependencies are available."""

    def test_audio_synthesis_with_deps(self, temp_workspace):
        """Test audio synthesis when FluidSynth is available."""
        # Full audio pipeline tests here
        pass
```

### CI/CD Configuration

Update CI workflows to test multiple installation modes:

```yaml
# .github/workflows/test.yml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
    install-mode:
      - "minimal"      # uv sync only
      - "audio-full"   # uv sync --extra audio-full

steps:
  - name: Install minimal dependencies
    if: matrix.install-mode == 'minimal'
    run: uv sync

  - name: Install full audio dependencies
    if: matrix.install-mode == 'audio-full'
    run: |
      # Install system dependencies
      sudo apt-get update
      sudo apt-get install -y fluidsynth libfluidsynth-dev ffmpeg
      # Install Python dependencies
      uv sync --extra audio-full

  - name: Run tests (minimal)
    if: matrix.install-mode == 'minimal'
    run: uv run pytest tests/test_cli_smoke.py::TestCLIMinimalInstallation -v

  - name: Run tests (full)
    if: matrix.install-mode == 'audio-full'
    run: uv run pytest -v
```

---

## Migration Plan

### Phase 1.1: Dependency Restructuring
**Timeline**: 1-2 days

1. **Update `pyproject.toml`**:
   - Move `pyfluidsynth` and `pydub` to optional dependencies
   - Create logical dependency groups
   - Update CLI script requirements

2. **Test minimal installation**:
   ```bash
   # Test on clean environment
   git clone <repo>
   cd musicgen
   uv sync
   uv run python -m musicgen.cli --help  # Should work
   ```

### Phase 1.2: Enhanced Fallback Implementation
**Timeline**: 2-3 days

1. **Update `synthesis.py`**:
   - Implement safe import patterns
   - Add informative warning messages
   - Return `None` for unavailable synthesis

2. **Update `cli.py`**:
   - Add export format validation
   - Implement graceful capability detection
   - Show helpful installation hints

3. **Update `io_files.py`**:
   - Handle audio export fallbacks
   - Skip unsupported format conversion

### Phase 1.3: Testing and Documentation
**Timeline**: 1-2 days

1. **Update test suite**:
   - Add minimal installation tests
   - Test capability detection
   - Test fallback behaviors

2. **Update documentation**:
   - Installation instructions for different modes
   - Troubleshooting guide
   - Feature matrix by installation type

3. **CI/CD updates**:
   - Test multiple installation modes
   - Verify minimal installation works

### Phase 1.4: User Experience Polish
**Timeline**: 1 day

1. **Improve warning messages**:
   - Clear installation instructions
   - Links to documentation
   - Capability status information

2. **Add info command enhancements**:
   ```bash
   uv run python -m musicgen.cli info
   # Should show:
   # - Installation mode (minimal/audio-synthesis/full)
   # - Available capabilities
   # - Missing dependencies with install instructions
   ```

---

## Success Validation

### Core Requirements Test
```bash
# Test 1: Minimal installation works everywhere
git clone <repo>
cd musicgen
uv sync
uv run python -m musicgen.cli generate test.mid --export midi
# ✅ Should succeed on any platform

# Test 2: Graceful fallback with warnings
uv run python -m musicgen.cli generate test.mid --export midi,wav,mp3
# ✅ Should warn about wav/mp3, produce MIDI

# Test 3: Upgrade path works
uv sync --extra audio-full
uv run python -m musicgen.cli generate test.mid --export midi,wav,mp3
# ✅ Should work if system deps available
```

### User Experience Test
```bash
# Test 4: Helpful information
uv run python -m musicgen.cli info
# ✅ Should show installation mode and upgrade options

# Test 5: Clear error messages
uv run python -m musicgen.cli generate test.mid --soundfont nonexistent.sf2
# ✅ Should show helpful SoundFont installation guide
```

---

## Implementation Notes

### Key Design Principles

1. **Fail Gracefully**: Never crash due to missing optional dependencies
2. **Inform Clearly**: Always explain what's missing and how to get it
3. **Progressive Enhancement**: Core features work everywhere, advanced features require more setup
4. **Developer Friendly**: Easy to test different installation modes

### Import Safety Pattern

Use this pattern throughout for optional dependencies:

```python
# Safe import pattern
try:
    import optional_dependency
    OPTIONAL_AVAILABLE = True
except ImportError as e:
    OPTIONAL_AVAILABLE = False
    _missing_dependencies["optional_dependency"] = str(e)

def feature_requiring_optional():
    if not OPTIONAL_AVAILABLE:
        _warn_missing_dependency("optional_dependency", "feature name")
        return None
    # Use optional_dependency...
```

### Configuration Validation

Enhance Pydantic models to validate against available capabilities:

```python
class MusicGenConfig(BaseModel):
    # ... existing fields ...

    @model_validator(mode='after')
    def validate_against_capabilities(self) -> Self:
        """Validate configuration against available capabilities."""
        available = get_audio_capabilities()

        # Validate export formats
        if self.export_formats:
            supported, unsupported = validate_export_formats(
                self.export_formats, available
            )
            if unsupported:
                warnings.warn(f"Unsupported export formats: {unsupported}")
                self.export_formats = supported or ["midi"]

        # Validate SoundFont requirement
        if ("wav" in self.export_formats or "mp3" in self.export_formats):
            if AudioCapability.AUDIO_SYNTHESIS not in available:
                warnings.warn("Audio synthesis not available")
                self.export_formats = ["midi"]

        return self
```

This comprehensive plan ensures that MusicGen becomes immediately accessible to new users while maintaining full functionality for those who need advanced audio features.