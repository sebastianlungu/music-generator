"""
Simple MIDI to WAV synthesis using mido and numpy.

This module provides a minimal synthesizer that can convert MIDI files to audio
without requiring pretty_midi or FluidSynth dependencies.
"""

import math
import wave

import mido
import numpy as np


def midi_note_to_freq(note_number):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note_number - 69) / 12.0))


def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    """Generate a sine wave for a given frequency and duration."""
    frames = int(duration * sample_rate)
    arr = np.zeros(frames)

    for i in range(frames):
        t = float(i) / sample_rate
        arr[i] = amplitude * math.sin(2 * math.pi * frequency * t)

    return arr


def synthesize_midi_file(midi_file_path, output_wav_path, sample_rate=44100):
    """
    Synthesize a MIDI file to WAV using simple sine waves.

    Args:
        midi_file_path: Path to input MIDI file
        output_wav_path: Path to output WAV file
        sample_rate: Audio sample rate
    """
    # Load MIDI file
    mid = mido.MidiFile(midi_file_path)

    # Calculate total duration in seconds
    total_ticks = 0
    for track in mid.tracks:
        track_ticks = sum(msg.time for msg in track)
        total_ticks = max(total_ticks, track_ticks)

    # Convert ticks to seconds
    tempo = 500000  # Default tempo (120 BPM)
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break

    seconds_per_tick = tempo / (1000000.0 * mid.ticks_per_beat)
    total_duration = total_ticks * seconds_per_tick

    if total_duration <= 0:
        total_duration = 4.0  # Default 4 seconds if no duration detected

    # Create audio buffer
    audio_frames = int(total_duration * sample_rate)
    audio_data = np.zeros(audio_frames)

    print(f"Synthesizing {total_duration:.2f} seconds of audio...")

    # Process each track
    for track_idx, track in enumerate(mid.tracks):
        print(f"Processing track {track_idx + 1}/{len(mid.tracks)}")

        # Track active notes and their start times
        active_notes = {}
        current_time = 0.0

        for msg in track:
            # Update current time
            current_time += msg.time * seconds_per_tick

            if msg.type == "note_on" and msg.velocity > 0:
                # Start note
                active_notes[msg.note] = current_time

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                # End note
                if msg.note in active_notes:
                    start_time = active_notes[msg.note]
                    duration = current_time - start_time

                    if duration > 0:
                        # Generate note audio
                        frequency = midi_note_to_freq(msg.note)
                        note_audio = generate_sine_wave(
                            frequency, duration, sample_rate, amplitude=0.1
                        )

                        # Add to audio buffer
                        start_frame = int(start_time * sample_rate)
                        end_frame = start_frame + len(note_audio)

                        if end_frame <= len(audio_data):
                            audio_data[start_frame:end_frame] += note_audio

                    del active_notes[msg.note]

    # Normalize audio to prevent clipping
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8

    # Convert to 16-bit integer
    audio_data_int = (audio_data * 32767).astype(np.int16)

    # Write WAV file
    with wave.open(output_wav_path, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data_int.tobytes())

    print(f"Audio synthesized to: {output_wav_path}")
    print(f"Duration: {total_duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")


if __name__ == "__main__":
    # Test with the melody file
    synthesize_midi_file("test_melody.mid", "test_melody.wav")
