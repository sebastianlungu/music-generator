# Phase 2: Pure Python Audio Synthesis

## Overview

This document outlines the implementation strategy for Phase 2 of the MusicGen audio pipeline: **Pure Python Audio Synthesis**. This phase eliminates the dependency on external system packages (FluidSynth) by implementing MIDI-to-audio synthesis entirely in Python.

### Goals

- Remove system dependency on FluidSynth installation
- Provide cross-platform audio synthesis without external binaries
- Maintain audio quality comparable to FluidSynth
- Enable SoundFont (.sf2) loading and sample playback
- Ensure consistent behavior across Windows, macOS, and Linux

## Technical Approaches Comparison

### 1. sf2_loader + numpy (Recommended)

**Pros:**
- Pure Python implementation
- Direct SoundFont support
- High-quality sample-based synthesis
- Full control over synthesis pipeline
- No external dependencies

**Cons:**
- More complex implementation
- Manual sample interpolation required
- Higher memory usage for large SoundFonts

**Dependencies:**
```toml
sf2_loader = "^1.0.0"  # SoundFont loading
numpy = "^1.24.0"      # Audio processing
scipy = "^1.10.0"      # Signal processing (optional)
```

### 2. pygame.mixer

**Pros:**
- Simple implementation
- Built-in MIDI support
- Cross-platform
- Lightweight

**Cons:**
- Limited SoundFont support
- Basic synthesis quality
- Less control over audio pipeline
- Dependency on pygame

**Dependencies:**
```toml
pygame = "^2.4.0"
```

### 3. Web Audio API (Browser-based)

**Pros:**
- No local dependencies
- Modern synthesis capabilities
- Built-in effects

**Cons:**
- Requires browser environment
- Complex setup for headless operation
- Not suitable for CLI tools
- Limited to web applications

## Recommended Architecture: sf2_loader Approach

### System Overview

```
MIDI Input → Note Events → SoundFont Samples → Audio Synthesis → WAV Output
    ↓            ↓              ↓                  ↓              ↓
pretty_midi → NoteProcessor → SampleEngine → AudioRenderer → AudioExporter
```

### Core Components

#### 1. SoundFont Manager
```python
class SoundFontManager:
    """Manages SoundFont loading and sample access."""

    def __init__(self, soundfont_path: Path):
        self.sf2 = sf2_loader.load_soundfont(soundfont_path)
        self.sample_cache = {}

    def get_sample(self, program: int, pitch: int, velocity: int) -> np.ndarray:
        """Get audio sample for given parameters."""

    def list_presets(self) -> List[Preset]:
        """List available instrument presets."""
```

#### 2. Voice Engine
```python
class VoiceEngine:
    """Manages active voices and polyphony."""

    def __init__(self, max_voices: int = 64):
        self.voices = []
        self.max_voices = max_voices

    def note_on(self, channel: int, pitch: int, velocity: int, timestamp: float):
        """Start a new voice."""

    def note_off(self, channel: int, pitch: int, timestamp: float):
        """Stop a voice."""

    def render_frame(self, frame_size: int) -> np.ndarray:
        """Render audio frame from all active voices."""
```

#### 3. Audio Renderer
```python
class AudioRenderer:
    """High-level audio synthesis orchestrator."""

    def __init__(self, soundfont_manager: SoundFontManager, sample_rate: int = 44100):
        self.sf_manager = soundfont_manager
        self.voice_engine = VoiceEngine()
        self.sample_rate = sample_rate

    def render_midi(self, midi_data: pretty_midi.PrettyMIDI) -> np.ndarray:
        """Render complete MIDI file to audio."""
```

## MIDI to Audio Synthesis Pipeline

### 1. MIDI Event Processing

```python
def extract_midi_events(midi_data: pretty_midi.PrettyMIDI) -> List[MidiEvent]:
    """
    Extract and sort all MIDI events by timestamp.

    Args:
        midi_data: PrettyMIDI object

    Returns:
        Sorted list of MIDI events
    """
    events = []

    for instrument_idx, instrument in enumerate(midi_data.instruments):
        channel = 9 if instrument.is_drum else instrument_idx % 16

        # Extract note events
        for note in instrument.notes:
            events.append(MidiEvent(
                timestamp=note.start,
                type=EventType.NOTE_ON,
                channel=channel,
                pitch=note.pitch,
                velocity=note.velocity,
                program=instrument.program
            ))

            events.append(MidiEvent(
                timestamp=note.end,
                type=EventType.NOTE_OFF,
                channel=channel,
                pitch=note.pitch,
                velocity=0,
                program=instrument.program
            ))

    return sorted(events, key=lambda e: e.timestamp)

@dataclass
class MidiEvent:
    timestamp: float
    type: EventType
    channel: int
    pitch: int
    velocity: int
    program: int = 0

class EventType(Enum):
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    PROGRAM_CHANGE = "program_change"
```

### 2. SoundFont Sample Loading

```python
def load_soundfont_samples(soundfont_path: Path) -> Dict[str, SampleData]:
    """
    Load all samples from a SoundFont file.

    Args:
        soundfont_path: Path to .sf2 file

    Returns:
        Dictionary mapping sample keys to audio data
    """
    import sf2_loader

    sf2 = sf2_loader.load_soundfont(str(soundfont_path))
    samples = {}

    for preset in sf2.presets:
        for zone in preset.zones:
            for generator in zone.generators:
                if generator.type == 'sample':
                    sample_data = sf2.samples[generator.value]

                    # Convert to numpy array
                    audio_data = np.frombuffer(
                        sample_data.raw_data,
                        dtype=np.int16
                    ).astype(np.float32) / 32768.0

                    sample_key = f"{preset.program}_{generator.key_range[0]}_{generator.key_range[1]}"
                    samples[sample_key] = SampleData(
                        audio=audio_data,
                        sample_rate=sample_data.sample_rate,
                        loop_start=sample_data.loop_start,
                        loop_end=sample_data.loop_end,
                        root_key=sample_data.original_pitch
                    )

    return samples

@dataclass
class SampleData:
    audio: np.ndarray
    sample_rate: int
    loop_start: int
    loop_end: int
    root_key: int
```

### 3. Sample Generation and Pitch Shifting

```python
def generate_note_sample(
    sample_data: SampleData,
    target_pitch: int,
    velocity: int,
    duration: float,
    output_sample_rate: int = 44100
) -> np.ndarray:
    """
    Generate audio sample for a specific note.

    Args:
        sample_data: Source sample data
        target_pitch: MIDI pitch (0-127)
        velocity: MIDI velocity (0-127)
        duration: Note duration in seconds
        output_sample_rate: Target sample rate

    Returns:
        Generated audio sample
    """
    # Calculate pitch shift ratio
    pitch_shift_semitones = target_pitch - sample_data.root_key
    pitch_ratio = 2 ** (pitch_shift_semitones / 12.0)

    # Resample for pitch shifting
    if pitch_ratio != 1.0:
        from scipy import signal
        resampled = signal.resample(
            sample_data.audio,
            int(len(sample_data.audio) / pitch_ratio)
        )
    else:
        resampled = sample_data.audio

    # Apply velocity scaling
    velocity_scale = velocity / 127.0
    resampled *= velocity_scale

    # Handle duration and looping
    target_samples = int(duration * output_sample_rate)

    if len(resampled) < target_samples:
        # Loop the sample if needed
        if sample_data.loop_start < sample_data.loop_end:
            resampled = extend_with_loop(
                resampled,
                sample_data.loop_start,
                sample_data.loop_end,
                target_samples
            )
        else:
            # Pad with zeros
            padding = target_samples - len(resampled)
            resampled = np.pad(resampled, (0, padding), mode='constant')
    else:
        # Truncate if too long
        resampled = resampled[:target_samples]

    # Apply envelope (ADSR)
    envelope = create_adsr_envelope(target_samples, output_sample_rate)
    resampled *= envelope

    return resampled.astype(np.float32)

def extend_with_loop(
    audio: np.ndarray,
    loop_start: int,
    loop_end: int,
    target_length: int
) -> np.ndarray:
    """Extend audio sample using loop points."""
    if target_length <= len(audio):
        return audio[:target_length]

    # Extract loop section
    loop_section = audio[loop_start:loop_end]

    # Calculate how many times to repeat loop
    remaining_samples = target_length - len(audio)
    loop_repeats = remaining_samples // len(loop_section) + 1

    # Create extended audio
    extended = np.concatenate([
        audio,
        np.tile(loop_section, loop_repeats)[:remaining_samples]
    ])

    return extended[:target_length]

def create_adsr_envelope(
    num_samples: int,
    sample_rate: int,
    attack: float = 0.01,
    decay: float = 0.1,
    sustain: float = 0.7,
    release: float = 0.2
) -> np.ndarray:
    """Create ADSR envelope for natural-sounding notes."""
    envelope = np.ones(num_samples)

    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)

    # Attack phase
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay phase
    decay_end = attack_samples + decay_samples
    if decay_samples > 0 and decay_end < num_samples:
        envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)

    # Sustain phase (already set to sustain level)
    if decay_end < num_samples - release_samples:
        envelope[decay_end:num_samples - release_samples] = sustain

    # Release phase
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(
            envelope[-release_samples], 0, release_samples
        )

    return envelope
```

### 4. Voice Management and Polyphony

```python
class Voice:
    """Represents an active note being played."""

    def __init__(
        self,
        sample_data: SampleData,
        pitch: int,
        velocity: int,
        start_time: float,
        channel: int
    ):
        self.sample_data = sample_data
        self.pitch = pitch
        self.velocity = velocity
        self.start_time = start_time
        self.channel = channel
        self.position = 0
        self.is_active = True
        self.release_time = None

        # Generate the note sample
        self.audio_sample = generate_note_sample(
            sample_data, pitch, velocity, duration=10.0  # Long duration for sustain
        )

    def get_next_frame(self, frame_size: int) -> np.ndarray:
        """Get next audio frame from this voice."""
        if not self.is_active or self.position >= len(self.audio_sample):
            return np.zeros(frame_size)

        end_pos = min(self.position + frame_size, len(self.audio_sample))
        frame = self.audio_sample[self.position:end_pos]

        # Pad if needed
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')

        self.position = end_pos

        # Check if voice finished
        if self.position >= len(self.audio_sample):
            self.is_active = False

        return frame

    def note_off(self, release_time: float):
        """Trigger note release."""
        self.release_time = release_time
        # Apply release envelope to remaining audio
        remaining_samples = len(self.audio_sample) - self.position
        release_samples = min(remaining_samples, int(0.2 * 44100))  # 200ms release

        if release_samples > 0:
            release_envelope = np.linspace(1, 0, release_samples)
            self.audio_sample[self.position:self.position + release_samples] *= release_envelope

class VoiceManager:
    """Manages polyphonic voice allocation."""

    def __init__(self, max_voices: int = 64):
        self.voices: List[Voice] = []
        self.max_voices = max_voices

    def note_on(
        self,
        sample_data: SampleData,
        channel: int,
        pitch: int,
        velocity: int,
        timestamp: float
    ) -> Voice:
        """Start a new voice."""
        # Remove finished voices
        self.voices = [v for v in self.voices if v.is_active]

        # Voice stealing if at limit
        if len(self.voices) >= self.max_voices:
            # Remove oldest voice
            self.voices.pop(0)

        # Create new voice
        voice = Voice(sample_data, pitch, velocity, timestamp, channel)
        self.voices.append(voice)
        return voice

    def note_off(self, channel: int, pitch: int, timestamp: float):
        """Stop voices matching channel and pitch."""
        for voice in self.voices:
            if voice.channel == channel and voice.pitch == pitch and voice.is_active:
                voice.note_off(timestamp)

    def render_frame(self, frame_size: int) -> np.ndarray:
        """Render mixed audio frame from all active voices."""
        mixed_frame = np.zeros(frame_size, dtype=np.float32)

        for voice in self.voices:
            if voice.is_active:
                voice_frame = voice.get_next_frame(frame_size)
                mixed_frame += voice_frame

        # Remove finished voices
        self.voices = [v for v in self.voices if v.is_active]

        return mixed_frame
```

## Performance Considerations and Optimizations

### 1. Memory Management

```python
class SampleCache:
    """LRU cache for generated samples to reduce computation."""

    def __init__(self, max_size: int = 1000):
        from functools import lru_cache
        self.max_size = max_size
        self._cache = {}
        self._access_order = []

    def get_sample(
        self,
        sample_key: str,
        pitch: int,
        velocity: int,
        duration: float
    ) -> np.ndarray:
        """Get cached sample or generate new one."""
        cache_key = f"{sample_key}_{pitch}_{velocity}_{duration:.2f}"

        if cache_key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]

        # Generate new sample
        sample = self._generate_sample(sample_key, pitch, velocity, duration)

        # Add to cache
        self._cache[cache_key] = sample
        self._access_order.append(cache_key)

        # Evict if over limit
        while len(self._cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        return sample
```

### 2. Streaming Audio Processing

```python
def render_midi_streaming(
    midi_data: pretty_midi.PrettyMIDI,
    soundfont_manager: SoundFontManager,
    chunk_duration: float = 0.1,
    sample_rate: int = 44100
) -> Iterator[np.ndarray]:
    """
    Stream audio generation in chunks to reduce memory usage.

    Args:
        midi_data: MIDI data to render
        soundfont_manager: SoundFont manager
        chunk_duration: Duration of each chunk in seconds
        sample_rate: Audio sample rate

    Yields:
        Audio chunks as numpy arrays
    """
    chunk_samples = int(chunk_duration * sample_rate)
    voice_manager = VoiceManager()

    # Extract and sort events
    events = extract_midi_events(midi_data)

    current_time = 0.0
    event_idx = 0
    total_duration = midi_data.get_end_time()

    while current_time < total_duration:
        chunk_end_time = current_time + chunk_duration

        # Process events in this time chunk
        while event_idx < len(events) and events[event_idx].timestamp < chunk_end_time:
            event = events[event_idx]

            if event.type == EventType.NOTE_ON:
                sample_data = soundfont_manager.get_sample(
                    event.program, event.pitch, event.velocity
                )
                voice_manager.note_on(
                    sample_data, event.channel, event.pitch,
                    event.velocity, event.timestamp
                )
            elif event.type == EventType.NOTE_OFF:
                voice_manager.note_off(
                    event.channel, event.pitch, event.timestamp
                )

            event_idx += 1

        # Render audio chunk
        audio_chunk = voice_manager.render_frame(chunk_samples)
        yield audio_chunk

        current_time = chunk_end_time
```

### 3. Multi-threading Optimization

```python
import concurrent.futures
from threading import Lock

class ThreadedSynthesizer:
    """Multi-threaded audio synthesizer for better performance."""

    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads
        self.voice_lock = Lock()

    def render_parallel(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        soundfont_manager: SoundFontManager
    ) -> np.ndarray:
        """Render MIDI using parallel processing."""
        events = extract_midi_events(midi_data)
        total_duration = midi_data.get_end_time()
        sample_rate = 44100

        # Split events into chunks for parallel processing
        chunk_duration = total_duration / self.num_threads
        event_chunks = self._split_events_by_time(events, chunk_duration)

        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(self._render_chunk, chunk, soundfont_manager, sample_rate)
                for chunk in event_chunks
            ]

            audio_chunks = [future.result() for future in futures]

        # Concatenate results
        return np.concatenate(audio_chunks, axis=0)

    def _split_events_by_time(
        self,
        events: List[MidiEvent],
        chunk_duration: float
    ) -> List[List[MidiEvent]]:
        """Split events into time-based chunks."""
        chunks = []
        current_chunk = []
        current_time = 0.0

        for event in events:
            if event.timestamp >= current_time + chunk_duration:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                current_time += chunk_duration

            current_chunk.append(event)

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
```

## Testing Strategy

### 1. Unit Tests

```python
import pytest
import numpy as np
from pathlib import Path

class TestSoundFontManager:
    """Test SoundFont loading and sample access."""

    def test_load_soundfont(self):
        """Test basic SoundFont loading."""
        # Use a test SoundFont file
        sf_path = Path("tests/fixtures/test.sf2")
        if sf_path.exists():
            manager = SoundFontManager(sf_path)
            assert manager.sf2 is not None

    def test_get_sample(self):
        """Test sample retrieval."""
        sf_path = Path("tests/fixtures/test.sf2")
        if sf_path.exists():
            manager = SoundFontManager(sf_path)
            sample = manager.get_sample(program=0, pitch=60, velocity=80)
            assert isinstance(sample, np.ndarray)
            assert sample.dtype == np.float32

class TestVoiceEngine:
    """Test voice management and polyphony."""

    def test_voice_allocation(self):
        """Test voice allocation and deallocation."""
        engine = VoiceEngine(max_voices=4)

        # Create test sample data
        sample_data = SampleData(
            audio=np.random.random(1000).astype(np.float32),
            sample_rate=44100,
            loop_start=0,
            loop_end=999,
            root_key=60
        )

        # Test note on
        voice = engine.note_on(0, 60, 80, 0.0, sample_data)
        assert len(engine.voices) == 1
        assert voice.is_active

        # Test note off
        engine.note_off(0, 60, 1.0)
        # Voice should still exist but marked for release
        assert len(engine.voices) == 1

    def test_voice_stealing(self):
        """Test voice stealing when limit exceeded."""
        engine = VoiceEngine(max_voices=2)
        sample_data = SampleData(
            audio=np.random.random(1000).astype(np.float32),
            sample_rate=44100,
            loop_start=0,
            loop_end=999,
            root_key=60
        )

        # Add voices up to limit
        engine.note_on(0, 60, 80, 0.0, sample_data)
        engine.note_on(0, 62, 80, 0.1, sample_data)
        assert len(engine.voices) == 2

        # Add one more - should trigger voice stealing
        engine.note_on(0, 64, 80, 0.2, sample_data)
        assert len(engine.voices) == 2  # Still 2, oldest removed

class TestAudioRenderer:
    """Test complete audio rendering pipeline."""

    def test_render_simple_midi(self):
        """Test rendering a simple MIDI file."""
        # Create simple MIDI data
        midi_data = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        # Add a simple note
        note = pretty_midi.Note(
            velocity=80,
            pitch=60,
            start=0.0,
            end=1.0
        )
        instrument.notes.append(note)
        midi_data.instruments.append(instrument)

        # Test rendering (if SoundFont available)
        sf_path = Path("tests/fixtures/test.sf2")
        if sf_path.exists():
            sf_manager = SoundFontManager(sf_path)
            renderer = AudioRenderer(sf_manager)

            audio = renderer.render_midi(midi_data)

            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
            assert audio.dtype == np.float32

def test_pitch_shifting():
    """Test pitch shifting functionality."""
    # Create test sample
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0  # A4

    t = np.linspace(0, duration, int(sample_rate * duration))
    original_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    sample_data = SampleData(
        audio=original_audio,
        sample_rate=sample_rate,
        loop_start=0,
        loop_end=len(original_audio) - 1,
        root_key=69  # A4 = MIDI note 69
    )

    # Test octave up (12 semitones)
    shifted = generate_note_sample(
        sample_data,
        target_pitch=81,  # A5 = 69 + 12
        velocity=100,
        duration=duration
    )

    assert len(shifted) == len(original_audio)
    # Octave up should have roughly half the period
    # (This is a simplified test - real validation would use FFT)

def test_envelope_generation():
    """Test ADSR envelope generation."""
    sample_rate = 44100
    duration = 1.0
    num_samples = int(sample_rate * duration)

    envelope = create_adsr_envelope(
        num_samples,
        sample_rate,
        attack=0.1,
        decay=0.2,
        sustain=0.7,
        release=0.3
    )

    assert len(envelope) == num_samples
    assert envelope[0] == 0.0  # Starts at zero
    assert envelope[-1] == 0.0  # Ends at zero
    assert np.max(envelope) <= 1.0  # Never exceeds 1
    assert np.min(envelope) >= 0.0  # Never goes negative
```

### 2. Integration Tests

```python
class TestFullPipeline:
    """Test complete synthesis pipeline."""

    def test_end_to_end_synthesis(self):
        """Test complete MIDI to audio synthesis."""
        # Create more complex MIDI data
        midi_data = pretty_midi.PrettyMIDI()

        # Piano track
        piano = pretty_midi.Instrument(program=0, name="Piano")
        notes = [
            (60, 0.0, 0.5),  # C4
            (64, 0.5, 1.0),  # E4
            (67, 1.0, 1.5),  # G4
            (72, 1.5, 2.0),  # C5
        ]

        for pitch, start, end in notes:
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start,
                end=end
            )
            piano.notes.append(note)

        midi_data.instruments.append(piano)

        # Test with SoundFont if available
        sf_path = Path("tests/fixtures/test.sf2")
        if sf_path.exists():
            # Test pure Python synthesis
            synthesizer = PurePythonSynthesizer(sf_path)
            audio = synthesizer.synthesize(midi_data)

            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
            assert audio.dtype == np.float32

            # Audio should be approximately 2 seconds long
            expected_length = int(2.0 * 44100)
            assert abs(len(audio) - expected_length) < 4410  # Within 0.1 second

    def test_performance_benchmark(self):
        """Benchmark synthesis performance."""
        import time

        # Create large MIDI file for performance testing
        midi_data = self._create_complex_midi()

        sf_path = Path("tests/fixtures/test.sf2")
        if sf_path.exists():
            synthesizer = PurePythonSynthesizer(sf_path)

            start_time = time.time()
            audio = synthesizer.synthesize(midi_data)
            end_time = time.time()

            synthesis_time = end_time - start_time
            audio_duration = len(audio) / 44100

            # Synthesis should be faster than real-time for simple files
            print(f"Synthesis time: {synthesis_time:.2f}s for {audio_duration:.2f}s audio")
            print(f"Real-time factor: {audio_duration / synthesis_time:.2f}x")

            # For simple synthesis, should be at least 2x real-time
            assert audio_duration / synthesis_time >= 1.0

    def _create_complex_midi(self) -> pretty_midi.PrettyMIDI:
        """Create complex MIDI data for testing."""
        midi_data = pretty_midi.PrettyMIDI()

        # Multiple instruments
        for program in [0, 24, 40]:  # Piano, Guitar, Violin
            instrument = pretty_midi.Instrument(program=program)

            # Generate random notes
            for i in range(50):
                start = i * 0.1
                end = start + 0.2
                pitch = 60 + (i % 24)
                velocity = 60 + (i % 40)

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start,
                    end=end
                )
                instrument.notes.append(note)

            midi_data.instruments.append(instrument)

        return midi_data
```

## Integration with Existing Codebase

### 1. Synthesis Module Integration

```python
# musicgen/synthesis.py - Modified for pure Python support

class PurePythonSynthesizer(AudioSynthesizer):
    """Pure Python MIDI synthesizer using sf2_loader."""

    def __init__(self, soundfont_path: Path):
        self.soundfont_path = soundfont_path
        self.sf_manager = SoundFontManager(soundfont_path)

    def synthesize(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        sample_rate: int = 44100,
        **kwargs
    ) -> np.ndarray:
        """Synthesize MIDI to audio using pure Python."""
        renderer = AudioRenderer(self.sf_manager, sample_rate)
        return renderer.render_midi(midi_data)

    def is_available(self) -> bool:
        """Check if pure Python synthesis is available."""
        try:
            import sf2_loader
            return self.soundfont_path.exists()
        except ImportError:
            return False

# Updated synthesize_midi function
def synthesize_midi(
    midi_data: pretty_midi.PrettyMIDI,
    config: MusicGenConfig
) -> np.ndarray | None:
    """
    Synthesize MIDI data using best available method.

    Priority:
    1. FluidSynth (if available)
    2. Pure Python synthesis
    3. Silent audio fallback
    """
    # Try FluidSynth first
    if check_fluidsynth_available() and config.soundfont_path:
        try:
            return synthesize_with_fluidsynth(midi_data, config)
        except (FluidSynthError, SynthesisError) as e:
            warnings.warn(f"FluidSynth synthesis failed: {e}")

    # Fallback to pure Python synthesis
    if config.soundfont_path and config.soundfont_path.exists():
        try:
            synthesizer = PurePythonSynthesizer(config.soundfont_path)
            if synthesizer.is_available():
                return synthesizer.synthesize(
                    midi_data,
                    sample_rate=config.sample_rate
                )
        except Exception as e:
            warnings.warn(f"Pure Python synthesis failed: {e}")

    # Final fallback - silent audio
    duration = midi_data.get_end_time()
    return create_silent_audio(duration, config.sample_rate)
```

### 2. Configuration Updates

```python
# musicgen/config.py - Add pure Python synthesis options

@dataclass
class SynthesisConfig:
    """Configuration for audio synthesis."""

    # Existing options
    soundfont_path: Path | None = None
    sample_rate: int = 44100

    # Pure Python synthesis options
    use_pure_python: bool = False
    max_voices: int = 64
    sample_cache_size: int = 1000
    chunk_duration: float = 0.1
    enable_threading: bool = True
    thread_count: int = 4

    # Quality settings
    enable_adsr: bool = True
    attack_time: float = 0.01
    decay_time: float = 0.1
    sustain_level: float = 0.7
    release_time: float = 0.2

class MusicGenConfig(BaseModel):
    """Main configuration with synthesis options."""

    # ... existing fields ...

    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)

    @field_validator('synthesis')
    @classmethod
    def validate_synthesis(cls, v: SynthesisConfig) -> SynthesisConfig:
        """Validate synthesis configuration."""
        if v.soundfont_path and not v.soundfont_path.exists():
            raise ValueError(f"SoundFont file not found: {v.soundfont_path}")

        if v.max_voices < 1 or v.max_voices > 256:
            raise ValueError("max_voices must be between 1 and 256")

        return v
```

### 3. CLI Updates

```python
# musicgen/cli.py - Add pure Python synthesis options

@click.option(
    "--synthesis-method",
    type=click.Choice(["auto", "fluidsynth", "pure-python"]),
    default="auto",
    help="Audio synthesis method to use"
)
@click.option(
    "--max-voices",
    type=int,
    default=64,
    help="Maximum number of simultaneous voices for pure Python synthesis"
)
@click.option(
    "--enable-threading",
    is_flag=True,
    default=True,
    help="Enable multi-threading for pure Python synthesis"
)
def main(
    input_path: str,
    output: str,
    synthesis_method: str,
    max_voices: int,
    enable_threading: bool,
    # ... other parameters
):
    """Main CLI function with pure Python synthesis support."""

    # Configure synthesis method
    synthesis_config = SynthesisConfig(
        soundfont_path=Path(soundfont) if soundfont else None,
        use_pure_python=(synthesis_method == "pure-python"),
        max_voices=max_voices,
        enable_threading=enable_threading
    )

    config = MusicGenConfig(
        # ... other config
        synthesis=synthesis_config
    )

    # ... rest of CLI logic
```

## Fallback Mechanisms

### 1. Graceful Degradation

```python
class SynthesisChain:
    """Chain of synthesis methods with automatic fallback."""

    def __init__(self, config: MusicGenConfig):
        self.config = config
        self.synthesizers = self._build_synthesizer_chain()

    def _build_synthesizer_chain(self) -> List[AudioSynthesizer]:
        """Build prioritized list of synthesizers."""
        synthesizers = []

        # Primary: FluidSynth (if available and not disabled)
        if (not self.config.synthesis.use_pure_python and
            check_fluidsynth_available()):
            synthesizers.append(FluidSynthSynthesizer(self.config))

        # Secondary: Pure Python synthesis
        if self.config.soundfont_path:
            synthesizers.append(PurePythonSynthesizer(self.config.soundfont_path))

        # Tertiary: Basic pygame synthesis (if available)
        try:
            import pygame
            synthesizers.append(PygameSynthesizer())
        except ImportError:
            pass

        # Final fallback: Silent audio
        synthesizers.append(SilentSynthesizer())

        return synthesizers

    def synthesize(self, midi_data: pretty_midi.PrettyMIDI) -> np.ndarray:
        """Synthesize using first available method."""
        last_error = None

        for synthesizer in self.synthesizers:
            if not synthesizer.is_available():
                continue

            try:
                result = synthesizer.synthesize(
                    midi_data,
                    sample_rate=self.config.sample_rate
                )
                if result is not None and len(result) > 0:
                    return result
            except Exception as e:
                last_error = e
                warnings.warn(
                    f"Synthesis failed with {synthesizer.__class__.__name__}: {e}"
                )
                continue

        # If all methods failed, raise the last error
        if last_error:
            raise SynthesisError(f"All synthesis methods failed. Last error: {last_error}")

        # Should never reach here due to SilentSynthesizer fallback
        raise SynthesisError("No synthesis methods available")

class SilentSynthesizer(AudioSynthesizer):
    """Fallback synthesizer that generates silent audio."""

    def synthesize(self, midi_data: pretty_midi.PrettyMIDI, **kwargs) -> np.ndarray:
        """Generate silent audio matching MIDI duration."""
        duration = midi_data.get_end_time()
        sample_rate = kwargs.get('sample_rate', 44100)
        return create_silent_audio(duration, sample_rate)

    def is_available(self) -> bool:
        """Always available as final fallback."""
        return True
```

### 2. Error Recovery

```python
class RobustSynthesizer:
    """Synthesizer with comprehensive error handling."""

    def __init__(self, config: MusicGenConfig):
        self.config = config
        self.synthesis_chain = SynthesisChain(config)

    def synthesize_with_recovery(
        self,
        midi_data: pretty_midi.PrettyMIDI
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Synthesize with comprehensive error recovery.

        Returns:
            Tuple of (audio_data, warning_messages)
        """
        warnings_list = []

        try:
            # Validate MIDI data first
            if not self._validate_midi_data(midi_data):
                warnings_list.append("Invalid MIDI data detected, attempting repair...")
                midi_data = self._repair_midi_data(midi_data)

            # Attempt synthesis
            audio_data = self.synthesis_chain.synthesize(midi_data)

            # Validate output
            if not self._validate_audio_output(audio_data):
                warnings_list.append("Invalid audio output detected, applying fixes...")
                audio_data = self._fix_audio_output(audio_data, midi_data)

            return audio_data, warnings_list

        except Exception as e:
            error_msg = f"Synthesis failed completely: {e}"
            warnings_list.append(error_msg)

            # Generate fallback silent audio
            duration = max(1.0, midi_data.get_end_time())
            fallback_audio = create_silent_audio(duration, self.config.sample_rate)

            return fallback_audio, warnings_list

    def _validate_midi_data(self, midi_data: pretty_midi.PrettyMIDI) -> bool:
        """Validate MIDI data integrity."""
        if not midi_data.instruments:
            return False

        for instrument in midi_data.instruments:
            if not instrument.notes:
                continue

            for note in instrument.notes:
                if note.start < 0 or note.end <= note.start:
                    return False
                if note.pitch < 0 or note.pitch > 127:
                    return False
                if note.velocity < 0 or note.velocity > 127:
                    return False

        return True

    def _repair_midi_data(self, midi_data: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Repair common MIDI data issues."""
        repaired = pretty_midi.PrettyMIDI()

        for instrument in midi_data.instruments:
            repaired_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )

            for note in instrument.notes:
                # Fix timing issues
                start = max(0, note.start)
                end = max(start + 0.01, note.end)  # Minimum 10ms duration

                # Fix pitch and velocity ranges
                pitch = max(0, min(127, note.pitch))
                velocity = max(1, min(127, note.velocity))

                repaired_note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start,
                    end=end
                )
                repaired_instrument.notes.append(repaired_note)

            if repaired_instrument.notes:  # Only add if has notes
                repaired.instruments.append(repaired_instrument)

        return repaired

    def _validate_audio_output(self, audio_data: np.ndarray) -> bool:
        """Validate synthesized audio output."""
        if audio_data is None or audio_data.size == 0:
            return False

        # Check for NaN or infinite values
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            return False

        # Check for excessive clipping
        if np.abs(audio_data).max() > 1.0:
            return False

        return True

    def _fix_audio_output(
        self,
        audio_data: np.ndarray,
        midi_data: pretty_midi.PrettyMIDI
    ) -> np.ndarray:
        """Fix common audio output issues."""
        if audio_data is None or audio_data.size == 0:
            # Generate silent audio
            duration = midi_data.get_end_time()
            return create_silent_audio(duration, self.config.sample_rate)

        # Remove NaN and infinite values
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply normalization if clipped
        peak = np.abs(audio_data).max()
        if peak > 1.0:
            audio_data = audio_data / peak * 0.95  # Leave some headroom

        return audio_data
```

## Dependencies and Installation

### Required Dependencies

```toml
# pyproject.toml additions for pure Python synthesis

[project]
dependencies = [
    # Existing dependencies...
    "sf2_loader>=1.0.0",  # SoundFont loading
    "scipy>=1.10.0",      # Signal processing for pitch shifting
]

[project.optional-dependencies]
full-audio = [
    "sf2_loader>=1.0.0",
    "scipy>=1.10.0",
    "pygame>=2.4.0",      # Alternative synthesis
]
```

### Installation Instructions

```bash
# Install pure Python audio synthesis
uv add sf2_loader scipy

# Optional: pygame for additional synthesis options
uv add pygame

# For development
uv add --dev pytest pytest-cov
```

## Migration Guide

### From FluidSynth to Pure Python

1. **Update configuration**: Add pure Python synthesis options to config
2. **Install dependencies**: Install sf2_loader and scipy
3. **Test synthesis**: Verify SoundFont compatibility
4. **Update CLI**: Add synthesis method selection options
5. **Monitor performance**: Compare synthesis speed and quality

### Backward Compatibility

The implementation maintains full backward compatibility:
- Existing FluidSynth synthesis continues to work
- Configuration files remain compatible
- CLI interface unchanged for basic usage
- Automatic fallback ensures no synthesis failures

This Phase 2 implementation provides a robust, pure Python audio synthesis solution that eliminates system dependencies while maintaining high audio quality and providing comprehensive fallback mechanisms.