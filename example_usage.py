#!/usr/bin/env python3
"""
Example usage of MusicGen - demonstrates core functionality.

This script shows how to:
1. Create a simple test MIDI file
2. Analyze it with MusicGen
3. Generate a new arrangement
4. Save the results

Run with: python example_usage.py
"""

from pathlib import Path

import mido


def create_sample_midi(file_path: Path) -> None:
    """Create a simple C major scale MIDI file for testing."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo (120 BPM)
    track.append(mido.MetaMessage("set_tempo", tempo=500000))

    # Add time signature
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))

    # C major scale
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5

    for _i, note in enumerate(notes):
        # Note on
        track.append(mido.Message("note_on", channel=0, note=note, velocity=80, time=0))
        # Note off after 480 ticks (quarter note at 480 PPQ)
        track.append(
            mido.Message("note_off", channel=0, note=note, velocity=80, time=480)
        )

    # Save the file
    mid.save(str(file_path))
    print(f"Created sample MIDI: {file_path}")


def main():
    """Main example workflow."""
    print("MusicGen Example Usage")
    print("=" * 30)

    # Create output directory
    output_dir = Path("./example_output")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Create a sample MIDI file
    input_midi = output_dir / "input_sample.mid"
    create_sample_midi(input_midi)

    # Step 2: Import MusicGen components
    from musicgen.config import ExportFormat, Instrument, MusicGenConfig
    from musicgen.orchestration import generate_arrangement

    # Step 3: Configure generation parameters
    config = MusicGenConfig(
        input_path=input_midi,
        output_dir=output_dir,
        duration_seconds=30,
        instruments=[Instrument.PIANO, Instrument.GUITAR],
        voices=2,
        style="classical",
        tempo_bpm=100,
        seed=42,
        export_formats=[ExportFormat.MIDI],  # MIDI only (no audio)
    )

    print("\nConfiguration:")
    print(f"  Input: {config.input_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  Duration: {config.duration_seconds}s")
    print(f"  Instruments: {[i.value for i in config.instruments]}")
    print(f"  Voices: {config.voices}")
    print(f"  Style: {config.style}")

    # Step 4: Generate arrangement
    print("\nGenerating arrangement...")
    try:
        result = generate_arrangement(config)

        if result.success:
            print("[SUCCESS] Generation completed!")
            print("\nGenerated files:")
            for file_type, file_path in result.output_files.items():
                if file_path.exists():
                    print(f"  {file_type}: {file_path}")

            print("\nAnalysis results:")
            analysis = result.analysis_result
            print(f"  Key: {analysis.key}")
            print(f"  Tempo: {analysis.tempo:.1f} BPM")
            print(
                f"  Time Signature: {analysis.time_signature[0]}/{analysis.time_signature[1]}"
            )
            print(f"  Duration: {analysis.duration_seconds:.1f}s")
            print(f"  Note Density: {analysis.note_density:.2f} notes/second")

        else:
            print(f"[ERROR] Generation failed: {result.error_message}")

    except Exception as e:
        print(f"[ERROR] Exception during generation: {e}")
        import traceback

        traceback.print_exc()

    print(f"\nExample complete! Check the '{output_dir}' directory for results.")


if __name__ == "__main__":
    main()
