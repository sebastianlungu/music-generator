# MusicGen Development Guide

Authoritative development guide for the **musicgen** project.

**Goal:** Build a small, typed, testable Python library + CLI that ingests MIDI files, analyzes them, generates new arrangements, and exports MIDI + audio.

---

## 📂 Project Structure

```
musicgen/
├── __init__.py
├── cli.py              # CLI entry
├── config.py           # Pydantic Config
├── io_files.py         # File discovery, MIDI/audio I/O
├── analysis.py         # Key/tempo/meter/structure, histograms
├── arrange.py          # Arrangement algorithms (rule-based, optional markov/nn)
├── orchestration.py    # High-level generate-from-analysis
└── synthesis.py        # Fluidsynth render + MP3 export
tests/
├── test_analysis.py
├── test_arrange.py
└── test_cli_smoke.py
pyproject.toml
README.md
```

---

## ⚙️ Toolchain & Commands

- **Python 3.10+**
- **uv** for everything (never `pip`):
  - Add dep: `uv add package`
  - Dev dep: `uv add --dev package`
  - Run tests: `uv run pytest`

### Format/lint:
```bash
uv run ruff format .
uv run ruff check . --fix
```

### Type check:
```bash
uv run pyright
```

### Run CLI:
```bash
uv run python -m musicgen.cli --help
```

---

## 🎨 Code Standards

- **Max function:** 50 lines (incl. comments/docstrings)
- **Max file:** 500 lines
- **Type hints everywhere**
- **Public functions require docstrings**
- **PEP8 naming** (snake_case, PascalCase, UPPER_SNAKE_CASE)
- **Pure functions** wherever possible, no hidden globals
- **Config/state passed explicitly** (via Pydantic config)

---

## 🧩 Module Responsibilities

- **`cli.py`** → CLI only (Typer/argparse), parses args, delegates to orchestration
- **`config.py`** → Pydantic config models, validation of CLI params
- **`io_files.py`** → file discovery, MIDI read/write, audio file write
- **`analysis.py`** → key/tempo/meter detection, sections, pitch histograms, density
- **`arrange.py`** → arrangement/generation algorithms (rule-based baseline; optional ML)
- **`orchestration.py`** → orchestrates analysis + arrangement + synthesis pipeline
- **`synthesis.py`** → fluidsynth rendering (SoundFont .sf2 → WAV), WAV→MP3 via pydub

### Tests

- **`test_analysis.py`** → ensure analysis returns key/tempo
- **`test_arrange.py`** → ensure arrangement produces valid notes in duration
- **`test_cli_smoke.py`** → run CLI end-to-end on tiny MIDI (skip audio in CI if ffmpeg absent)

---

## ✅ Golden Path for Changes

### 1. Design
- Define typed function signature & config model
- Place in correct module; keep boundaries clean

### 2. Tests first
- Unit tests for analysis/arrangement
- Integration smoke test for CLI

### 3. Implementation
- Keep I/O at edges
- Analysis/arrange logic should be pure
- Explicit error handling (no silent excepts)

### 4. Quality gate
```bash
uv run ruff format . && uv run ruff check . --fix
uv run pyright
uv run pytest
```

---

## 🔌 Allowed Libraries

- **MIDI I/O & analysis:** pretty_midi, mido, music21 (light use), numpy
- **Symbolic modeling:** rule-based + markovify, or tiny PyTorch model (optional)
- **Synthesis:** pyfluidsynth with user-supplied .sf2 SoundFont
- **Audio post:** pydub (requires ffmpeg)
- **CLI:** typer (or argparse fallback)
- **Config:** pydantic

---

## 🎯 CLI & API Inputs

- `--input` path (file or directory)
- `--output` directory (default ./out)
- `--duration-seconds` hard cap
- `--instruments` CSV (piano,guitar,voice)
- `--voices` int (1–8)
- `--style` free text
- `--tempo-bpm` or `--tempo-range` 90:120
- `--key` (optional, else detect)
- `--seed` int
- `--soundfont` path to .sf2
- `--export` formats (midi,wav,mp3; default all)

---

## 📦 Outputs

For each input (or aggregate folder), write to `out/<slug>/`:

- `analysis.json` (key, tempo, meter, pitch hist, density, duration)
- `generated.mid`
- `render.wav` (44.1 kHz, 24-bit)
- `render.mp3` (CBR 192 kbps)
- `report.txt` (parameters + short rationale)

---

## 🚫 What NOT to Do

- **No global state**
- **No silent failures** — errors must be actionable
- **No DAW-grade mixing/mastering** (out of scope)
- **No cloud services** (local only, unless trivial opt-in)
- **PostgreSQL/dbs irrelevant here** — do not introduce

---

## 📑 Implementation Roadmap

1. Scaffold pyproject.toml, modules, minimal CLI
2. Implement analysis.py (key/tempo/meter)
3. Add baseline arrange.py (rule-based piano sketch)
4. Add synthesis.py with SoundFont render + MP3 export
5. Write integration CLI test
6. Iterate: multi-voice, style options, analysis JSON, rationale report

---

## 🔑 Quick Commands Reference

```bash
# Run formatting & lint
uv run ruff format . && uv run ruff check . --fix

# Type check
uv run pyright

# Run tests
uv run pytest

# Run CLI (example)
uv run python -m musicgen.cli \
  --input ./midi/example.mid \
  --output ./out \
  --duration-seconds 120 \
  --instruments "piano,guitar" \
  --voices 2 \
  --style "ambient pop" \
  --tempo-range 90:110 \
  --soundfont ./assets/GeneralUser.sf2 \
  --export midi,wav,mp3
```

---

**Keep this file concise.** Push deep details to per-module READMEs if needed.

**North star:** typed, small, test-backed functions in clean modules.

---

*Would you like me to also generate a **starter `pyproject.toml` + stub modules** (`cli.py`, `analysis.py`, etc.) so you can immediately run `uv run python -m musicgen.cli` and see a "hello world" analysis on a sample MIDI?*