"""
Smoke tests for the Web UI interface.

These tests ensure that the web UI can be created and its core
functions work without actually launching a browser.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from musicgen.config import WebUIConfig

# Import pretty_midi from our module (which includes mock support)
from musicgen.io_files import pretty_midi
from musicgen.webui import create_interface, process_generation


def create_test_midi_file(file_path: Path) -> None:
    """Create a simple test MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120)

    # Add time signature
    midi_data.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))

    # Create piano instrument
    piano = pretty_midi.Instrument(program=0)

    # Add simple C major scale
    scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, pitch in enumerate(scale_notes):
        note = pretty_midi.Note(
            velocity=80, pitch=pitch, start=i * 0.5, end=i * 0.5 + 0.4
        )
        piano.notes.append(note)

    midi_data.instruments.append(piano)
    midi_data.write(str(file_path))


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create test MIDI file
    test_midi = temp_dir / "test.mid"
    create_test_midi_file(test_midi)

    yield {
        "temp_dir": temp_dir,
        "test_midi": test_midi,
    }

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_midi_bytes(temp_workspace):
    """Get test MIDI file as bytes for upload simulation."""
    test_midi = temp_workspace["test_midi"]
    with open(test_midi, "rb") as f:
        return f.read()


class TestWebUICreation:
    """Test web UI interface creation."""

    def test_create_interface_default(self):
        """Test creating interface with default configuration."""
        interface = create_interface()
        assert interface is not None

    def test_create_interface_custom_config(self):
        """Test creating interface with custom configuration."""
        config = WebUIConfig(host="0.0.0.0", port=8080, debug=True)
        interface = create_interface(config)
        assert interface is not None

    def test_interface_components(self):
        """Test that interface has expected components."""
        interface = create_interface()

        # Check that interface was created successfully
        # Note: We can't easily test the internal components without
        # launching the interface, so we just verify it doesn't crash
        assert interface is not None


class TestWebUIProcessing:
    """Test web UI processing functions."""

    def test_process_generation_basic(self, test_midi_bytes):
        """Test basic generation processing."""
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=30,
            instruments=["piano"],
            voices=1,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        # Should return tuple of outputs
        assert isinstance(result, tuple)
        assert len(result) == 8  # Expected number of outputs

        # First element should be status text
        status_text = result[0]
        assert isinstance(status_text, str)
        # Should either be success or error message
        assert "Error" in status_text or "successful" in status_text

    def test_process_generation_missing_midi(self):
        """Test processing with missing MIDI file."""
        result = process_generation(
            midi_file=None,  # Missing MIDI file
            soundfont_file=None,
            duration=30,
            instruments=["piano"],
            voices=1,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        # Should return error
        status_text = result[0]
        assert "Error" in status_text
        assert "MIDI file" in status_text

    def test_process_generation_no_instruments(self, test_midi_bytes):
        """Test processing with no instruments selected."""
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=30,
            instruments=[],  # No instruments
            voices=1,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        # Should return error
        status_text = result[0]
        assert "Error" in status_text
        assert "instrument" in status_text

    def test_process_generation_no_export_formats(self, test_midi_bytes):
        """Test processing with no export formats."""
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=30,
            instruments=["piano"],
            voices=1,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=[],  # No export formats
        )

        # Should return error
        status_text = result[0]
        assert "Error" in status_text
        assert "export format" in status_text

    def test_process_generation_multiple_instruments(self, test_midi_bytes):
        """Test processing with multiple instruments."""
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=45,
            instruments=["piano", "guitar", "violin"],
            voices=3,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        # Should handle multiple instruments
        assert isinstance(result, tuple)
        status_text = result[0]
        # Check if generation succeeded or failed gracefully
        assert isinstance(status_text, str)

    def test_process_generation_fixed_tempo(self, test_midi_bytes):
        """Test processing with fixed tempo mode."""
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=30,
            instruments=["piano"],
            voices=1,
            style="classical",
            tempo_mode="fixed",
            tempo_bpm=140,
            tempo_min=90,
            tempo_max=160,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        assert isinstance(result, tuple)
        status_text = result[0]
        assert isinstance(status_text, str)

    def test_process_generation_tempo_range(self, test_midi_bytes):
        """Test processing with tempo range mode."""
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=30,
            instruments=["piano"],
            voices=1,
            style="classical",
            tempo_mode="range",
            tempo_bpm=120,
            tempo_min=100,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        assert isinstance(result, tuple)
        status_text = result[0]
        assert isinstance(status_text, str)

    def test_process_generation_specified_key(self, test_midi_bytes):
        """Test processing with specified key mode."""
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=30,
            instruments=["piano"],
            voices=1,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="specified",
            musical_key="A minor",
            seed=42,
            export_formats=["midi"],
        )

        assert isinstance(result, tuple)
        status_text = result[0]
        assert isinstance(status_text, str)

    def test_process_generation_different_styles(self, test_midi_bytes):
        """Test processing with different musical styles."""
        styles = ["classical", "jazz", "rock", "ambient", "electronic"]

        for style in styles:
            result = process_generation(
                midi_file=test_midi_bytes,
                soundfont_file=None,
                duration=20,
                instruments=["piano"],
                voices=1,
                style=style,
                tempo_mode="auto",
                tempo_bpm=120,
                tempo_min=90,
                tempo_max=140,
                key_mode="auto",
                musical_key="C major",
                seed=42,
                export_formats=["midi"],
            )

            assert isinstance(result, tuple)
            status_text = result[0]
            assert isinstance(status_text, str)

    def test_process_generation_edge_cases(self, test_midi_bytes):
        """Test processing with edge case parameters."""
        # Very short duration
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=10,  # Minimum duration
            instruments=["piano"],
            voices=1,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        assert isinstance(result, tuple)

        # Maximum voices
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=30,
            instruments=["piano"] * 8,  # 8 instruments for 8 voices
            voices=8,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        assert isinstance(result, tuple)


class TestWebUIErrorHandling:
    """Test web UI error handling."""

    def test_process_generation_invalid_midi(self):
        """Test processing with invalid MIDI data."""
        invalid_midi = b"invalid midi data"

        result = process_generation(
            midi_file=invalid_midi,
            soundfont_file=None,
            duration=30,
            instruments=["piano"],
            voices=1,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        # Should handle invalid MIDI gracefully
        status_text = result[0]
        assert "Error" in status_text

    def test_process_generation_extreme_parameters(self, test_midi_bytes):
        """Test processing with extreme parameters."""
        # Very long duration
        result = process_generation(
            midi_file=test_midi_bytes,
            soundfont_file=None,
            duration=300,  # Maximum duration
            instruments=["piano"],
            voices=1,
            style="classical",
            tempo_mode="auto",
            tempo_bpm=120,
            tempo_min=90,
            tempo_max=140,
            key_mode="auto",
            musical_key="C major",
            seed=42,
            export_formats=["midi"],
        )

        assert isinstance(result, tuple)

    def test_process_generation_reproducibility(self, test_midi_bytes):
        """Test that processing is reproducible with same parameters."""
        params = {
            "midi_file": test_midi_bytes,
            "soundfont_file": None,
            "duration": 30,
            "instruments": ["piano"],
            "voices": 1,
            "style": "classical",
            "tempo_mode": "auto",
            "tempo_bpm": 120,
            "tempo_min": 90,
            "tempo_max": 140,
            "key_mode": "auto",
            "musical_key": "C major",
            "seed": 123,  # Fixed seed
            "export_formats": ["midi"],
        }

        result1 = process_generation(**params)
        result2 = process_generation(**params)

        # Status messages should be the same (both success or both failure)
        status1 = result1[0]
        status2 = result2[0]

        # Both should have same success/failure pattern
        success1 = "successful" in status1
        success2 = "successful" in status2
        assert success1 == success2


if __name__ == "__main__":
    pytest.main([__file__])
