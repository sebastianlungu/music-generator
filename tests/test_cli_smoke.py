"""
Smoke tests for the CLI interface.

These tests ensure that the CLI can run end-to-end without crashing,
creating expected output files, and handling basic error cases.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import pretty_midi

from musicgen.cli import main
from musicgen.config import ExportFormat


def create_test_midi_file(file_path: Path) -> None:
    """Create a simple test MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120)

    # Add time signature
    midi_data.time_signature_changes.append(
        pretty_midi.TimeSignature(4, 4, 0)
    )

    # Create piano instrument
    piano = pretty_midi.Instrument(program=0)

    # Add simple C major scale
    scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, pitch in enumerate(scale_notes):
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=i * 0.5,
            end=i * 0.5 + 0.4
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

    # Create output directory
    output_dir = temp_dir / "output"
    output_dir.mkdir()

    yield {
        "temp_dir": temp_dir,
        "test_midi": test_midi,
        "output_dir": output_dir
    }

    # Cleanup
    shutil.rmtree(temp_dir)


class TestCLISmoke:
    """Smoke tests for CLI functionality."""

    def test_cli_generate_basic(self, temp_workspace):
        """Test basic CLI generation command."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        # Mock sys.argv to simulate CLI call
        test_args = [
            "musicgen",
            "generate",
            str(test_midi),
            "--output", str(output_dir),
            "--duration-seconds", "10",
            "--instruments", "piano",
            "--voices", "1",
            "--export", "midi",  # Only MIDI to avoid audio dependencies
        ]

        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit as e:
                # CLI should exit with code 0 on success
                assert e.code == 0 or e.code is None

        # Check that output files were created
        output_files = list(output_dir.glob("**/*"))
        assert len(output_files) > 0

        # Check for expected files
        midi_files = list(output_dir.glob("**/*.mid"))
        json_files = list(output_dir.glob("**/*.json"))
        txt_files = list(output_dir.glob("**/*.txt"))

        assert len(midi_files) > 0, "No MIDI files generated"
        assert len(json_files) > 0, "No analysis JSON generated"
        assert len(txt_files) > 0, "No report file generated"

    def test_cli_generate_multiple_instruments(self, temp_workspace):
        """Test CLI with multiple instruments."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        test_args = [
            "musicgen",
            "generate",
            str(test_midi),
            "--output", str(output_dir),
            "--duration-seconds", "15",
            "--instruments", "piano,guitar,violin",
            "--voices", "3",
            "--style", "classical",
            "--export", "midi",
        ]

        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0 or e.code is None

        # Should generate files successfully
        midi_files = list(output_dir.glob("**/*.mid"))
        assert len(midi_files) > 0

    def test_cli_generate_with_tempo_range(self, temp_workspace):
        """Test CLI with tempo range."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        test_args = [
            "musicgen",
            "generate",
            str(test_midi),
            "--output", str(output_dir),
            "--tempo-range", "100:140",
            "--export", "midi",
        ]

        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0 or e.code is None

    def test_cli_generate_with_key(self, temp_workspace):
        """Test CLI with specific key."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        test_args = [
            "musicgen",
            "generate",
            str(test_midi),
            "--output", str(output_dir),
            "--key", "A minor",
            "--export", "midi",
        ]

        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0 or e.code is None

    def test_cli_analyze_command(self, temp_workspace):
        """Test CLI analyze command."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]
        output_json = output_dir / "analysis.json"

        test_args = [
            "musicgen",
            "analyze",
            str(test_midi),
            "--output", str(output_json),
        ]

        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0 or e.code is None

        # Check that analysis file was created
        assert output_json.exists(), "Analysis JSON file not created"

        # Check that JSON is valid
        import json
        with open(output_json) as f:
            analysis_data = json.load(f)

        assert "key" in analysis_data
        assert "tempo" in analysis_data
        assert "time_signature" in analysis_data

    def test_cli_info_command(self):
        """Test CLI info command."""
        test_args = ["musicgen", "info"]

        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0 or e.code is None

    def test_cli_error_missing_input(self, temp_workspace):
        """Test CLI error handling for missing input file."""
        output_dir = temp_workspace["output_dir"]
        nonexistent_file = temp_workspace["temp_dir"] / "nonexistent.mid"

        test_args = [
            "musicgen",
            "generate",
            str(nonexistent_file),
            "--output", str(output_dir),
        ]

        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Should exit with error code
            assert exc_info.value.code != 0

    def test_cli_error_invalid_instruments(self, temp_workspace):
        """Test CLI with invalid instrument names."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        test_args = [
            "musicgen",
            "generate",
            str(test_midi),
            "--output", str(output_dir),
            "--instruments", "invalid_instrument,another_invalid",
            "--export", "midi",
        ]

        with patch("sys.argv", test_args):
            # Should still work but substitute valid instruments
            try:
                main()
            except SystemExit as e:
                # Might succeed with warnings, or fail with validation error
                pass

    def test_cli_error_invalid_tempo_range(self, temp_workspace):
        """Test CLI with invalid tempo range."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        test_args = [
            "musicgen",
            "generate",
            str(test_midi),
            "--output", str(output_dir),
            "--tempo-range", "invalid:range",
        ]

        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0

    def test_cli_verbose_mode(self, temp_workspace):
        """Test CLI with verbose output."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        test_args = [
            "musicgen",
            "generate",
            str(test_midi),
            "--output", str(output_dir),
            "--duration-seconds", "5",
            "--export", "midi",
            "--verbose",
        ]

        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0 or e.code is None

    def test_cli_batch_mode(self, temp_workspace):
        """Test CLI batch processing mode."""
        temp_dir = temp_workspace["temp_dir"]
        output_dir = temp_workspace["output_dir"]

        # Create additional MIDI files
        for i in range(3):
            midi_file = temp_dir / f"test_{i}.mid"
            create_test_midi_file(midi_file)

        test_args = [
            "musicgen",
            "generate",
            str(temp_dir),
            "--output", str(output_dir),
            "--batch",
            "--duration-seconds", "5",
            "--export", "midi",
        ]

        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0 or e.code is None

        # Should have created output for multiple files
        output_subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        assert len(output_subdirs) >= 3, "Batch processing didn't create expected outputs"


class TestCLIParameterParsing:
    """Test CLI parameter parsing and validation."""

    def test_parse_instruments_valid(self, temp_workspace):
        """Test parsing of valid instrument names."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        # Test various valid instrument combinations
        valid_instruments = [
            "piano",
            "piano,guitar",
            "piano,guitar,violin",
            "strings,brass,woodwinds",
        ]

        for instruments_str in valid_instruments:
            test_args = [
                "musicgen",
                "generate",
                str(test_midi),
                "--output", str(output_dir),
                "--instruments", instruments_str,
                "--export", "midi",
            ]

            with patch("sys.argv", test_args):
                try:
                    main()
                except SystemExit as e:
                    # Should succeed for all valid combinations
                    assert e.code == 0 or e.code is None

    def test_parse_export_formats(self, temp_workspace):
        """Test parsing of export format combinations."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        # Test different export format combinations
        format_combinations = [
            "midi",
            "midi,wav",  # Note: WAV might not work without SoundFont
            "midi,wav,mp3",
        ]

        for formats_str in format_combinations:
            test_args = [
                "musicgen",
                "generate",
                str(test_midi),
                "--output", str(output_dir),
                "--export", formats_str,
            ]

            with patch("sys.argv", test_args):
                try:
                    main()
                except SystemExit as e:
                    # MIDI should always work
                    if formats_str == "midi":
                        assert e.code == 0 or e.code is None

    def test_duration_constraints(self, temp_workspace):
        """Test duration parameter constraints."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        # Test various durations
        durations = [10, 30, 60, 120, 300]  # Valid range

        for duration in durations:
            test_args = [
                "musicgen",
                "generate",
                str(test_midi),
                "--output", str(output_dir),
                "--duration-seconds", str(duration),
                "--export", "midi",
            ]

            with patch("sys.argv", test_args):
                try:
                    main()
                except SystemExit as e:
                    assert e.code == 0 or e.code is None

    def test_voices_constraints(self, temp_workspace):
        """Test voices parameter constraints."""
        test_midi = temp_workspace["test_midi"]
        output_dir = temp_workspace["output_dir"]

        # Test various voice counts
        voice_counts = [1, 2, 4, 8]  # Valid range

        for voices in voice_counts:
            # Need enough instruments for voices
            instruments = ["piano"] * voices

            test_args = [
                "musicgen",
                "generate",
                str(test_midi),
                "--output", str(output_dir),
                "--voices", str(voices),
                "--instruments", ",".join(instruments),
                "--export", "midi",
            ]

            with patch("sys.argv", test_args):
                try:
                    main()
                except SystemExit as e:
                    assert e.code == 0 or e.code is None


if __name__ == "__main__":
    pytest.main([__file__])