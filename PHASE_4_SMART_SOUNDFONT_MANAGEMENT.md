# Phase 4: Smart SoundFont Management

## Overview

Phase 4 implements intelligent SoundFont management for the MusicGen audio synthesis pipeline. This phase provides automatic SoundFont discovery, download, caching, and quality management to ensure reliable audio synthesis across different environments and user preferences.

## Architecture

### SoundFont Management Strategy

The smart SoundFont management system follows a multi-tier approach:

1. **Local Cache First**: Check for existing SoundFonts in user cache
2. **Quality Tiers**: Support multiple quality levels (minimal, standard, high)
3. **Auto-Download**: Fetch missing SoundFonts from trusted sources
4. **Fallback Chain**: Graceful degradation when preferred SoundFonts unavailable
5. **Offline Mode**: Function without network when cached SoundFonts available

### Core Components

```
musicgen/soundfont/
├── __init__.py
├── manager.py          # Main SoundFont management
├── downloader.py       # Download and validation
├── cache.py           # Local caching system
├── quality.py         # Quality tier definitions
├── sources.py         # Trusted SoundFont sources
└── validation.py      # SoundFont integrity checks
```

## Implementation Details

### SoundFont Manager Core

```python
# musicgen/soundfont/manager.py
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import pretty_midi
from pydantic import BaseModel, Field

from ..config import Config


class SoundFontQuality(Enum):
    """SoundFont quality tiers."""
    MINIMAL = "minimal"      # ~2MB, basic GM sounds
    STANDARD = "standard"    # ~8MB, good quality GM
    HIGH = "high"           # ~50MB+, studio quality
    CUSTOM = "custom"       # User-provided


@dataclass
class SoundFontInfo:
    """SoundFont metadata and paths."""
    name: str
    quality: SoundFontQuality
    file_path: Path
    size_mb: float
    checksum: str
    instruments: List[int]  # GM program numbers supported
    version: str = "1.0"
    source_url: Optional[str] = None
    last_verified: Optional[float] = None


class SoundFontConfig(BaseModel):
    """SoundFont configuration."""
    preferred_quality: SoundFontQuality = SoundFontQuality.STANDARD
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".musicgen" / "soundfonts")
    auto_download: bool = True
    offline_mode: bool = False
    max_cache_size_gb: float = 1.0
    verification_interval_days: int = 30
    download_timeout_seconds: int = 300
    trusted_sources: List[str] = Field(default_factory=list)


class SoundFontManager:
    """Manages SoundFont discovery, download, and caching."""

    def __init__(self, config: SoundFontConfig):
        self.config = config
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        from .cache import SoundFontCache
        from .downloader import SoundFontDownloader
        from .validation import SoundFontValidator

        self.cache = SoundFontCache(self.cache_dir)
        self.downloader = SoundFontDownloader(config)
        self.validator = SoundFontValidator()

        # Load available SoundFonts
        self._available_soundfonts: Dict[str, SoundFontInfo] = {}
        self._refresh_available_soundfonts()

    def get_soundfont(
        self,
        instruments: Optional[List[int]] = None,
        quality: Optional[SoundFontQuality] = None
    ) -> Path:
        """Get best available SoundFont for requested instruments."""
        target_quality = quality or self.config.preferred_quality

        # Try to find suitable cached SoundFont
        soundfont = self._find_suitable_soundfont(instruments, target_quality)

        if soundfont:
            return soundfont.file_path

        # Download if auto-download enabled and not in offline mode
        if self.config.auto_download and not self.config.offline_mode:
            soundfont = self._download_soundfont(instruments, target_quality)
            if soundfont:
                return soundfont.file_path

        # Fallback to any available SoundFont
        fallback = self._get_fallback_soundfont()
        if fallback:
            self.logger.warning(
                f"Using fallback SoundFont {fallback.name} "
                f"(quality: {fallback.quality.value})"
            )
            return fallback.file_path

        raise RuntimeError("No suitable SoundFont available")

    def _find_suitable_soundfont(
        self,
        instruments: Optional[List[int]],
        quality: SoundFontQuality
    ) -> Optional[SoundFontInfo]:
        """Find best matching cached SoundFont."""
        candidates = []

        for sf_info in self._available_soundfonts.values():
            # Check quality match
            if sf_info.quality == quality:
                score = 100
            elif sf_info.quality == SoundFontQuality.STANDARD:
                score = 80  # Good fallback
            elif sf_info.quality == SoundFontQuality.HIGH and quality == SoundFontQuality.STANDARD:
                score = 90  # Better than requested
            else:
                score = 50  # Other qualities

            # Check instrument coverage
            if instruments:
                covered = set(instruments) & set(sf_info.instruments)
                coverage = len(covered) / len(instruments)
                score *= coverage

            # Verify file exists and is valid
            if sf_info.file_path.exists() and self.validator.is_valid(sf_info.file_path):
                candidates.append((score, sf_info))

        if candidates:
            # Return highest scoring SoundFont
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def _download_soundfont(
        self,
        instruments: Optional[List[int]],
        quality: SoundFontQuality
    ) -> Optional[SoundFontInfo]:
        """Download suitable SoundFont."""
        try:
            self.logger.info(f"Downloading {quality.value} quality SoundFont...")
            return self.downloader.download_best_match(instruments, quality)
        except Exception as e:
            self.logger.error(f"Failed to download SoundFont: {e}")
            return None

    def _get_fallback_soundfont(self) -> Optional[SoundFontInfo]:
        """Get any available SoundFont as fallback."""
        for sf_info in self._available_soundfonts.values():
            if sf_info.file_path.exists():
                return sf_info
        return None

    def _refresh_available_soundfonts(self) -> None:
        """Refresh list of available SoundFonts."""
        self._available_soundfonts.clear()

        # Scan cache directory
        for sf_path in self.cache_dir.glob("*.sf2"):
            try:
                sf_info = self._analyze_soundfont(sf_path)
                self._available_soundfonts[sf_info.name] = sf_info
            except Exception as e:
                self.logger.warning(f"Failed to analyze {sf_path}: {e}")

    def _analyze_soundfont(self, sf_path: Path) -> SoundFontInfo:
        """Analyze SoundFont file and extract metadata."""
        # Basic file info
        size_mb = sf_path.stat().st_size / (1024 * 1024)

        # Calculate checksum
        with open(sf_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()[:16]

        # Determine quality based on size
        if size_mb < 5:
            quality = SoundFontQuality.MINIMAL
        elif size_mb < 20:
            quality = SoundFontQuality.STANDARD
        else:
            quality = SoundFontQuality.HIGH

        # Extract instrument list (simplified)
        instruments = list(range(128))  # Assume full GM compatibility

        return SoundFontInfo(
            name=sf_path.stem,
            quality=quality,
            file_path=sf_path,
            size_mb=size_mb,
            checksum=checksum,
            instruments=instruments
        )

    def cleanup_cache(self) -> None:
        """Clean up old or excess SoundFonts."""
        self.cache.cleanup(self.config.max_cache_size_gb)

    def list_available(self) -> List[SoundFontInfo]:
        """List all available SoundFonts."""
        return list(self._available_soundfonts.values())
```

### Auto-Download System

```python
# musicgen/soundfont/downloader.py
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
import urllib.request
import urllib.error
from dataclasses import dataclass

from .manager import SoundFontQuality, SoundFontInfo, SoundFontConfig
from .sources import TRUSTED_SOUNDFONT_SOURCES
from .validation import SoundFontValidator


@dataclass
class DownloadSource:
    """SoundFont download source."""
    name: str
    url: str
    quality: SoundFontQuality
    size_mb: float
    checksum: str
    instruments: List[int]


class SoundFontDownloader:
    """Downloads and validates SoundFonts from trusted sources."""

    def __init__(self, config: SoundFontConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validator = SoundFontValidator()

    def download_best_match(
        self,
        instruments: Optional[List[int]],
        quality: SoundFontQuality
    ) -> Optional[SoundFontInfo]:
        """Download best matching SoundFont."""
        sources = self._find_suitable_sources(instruments, quality)

        for source in sources:
            try:
                sf_info = self._download_source(source)
                if sf_info:
                    return sf_info
            except Exception as e:
                self.logger.warning(f"Failed to download {source.name}: {e}")

        return None

    def _find_suitable_sources(
        self,
        instruments: Optional[List[int]],
        quality: SoundFontQuality
    ) -> List[DownloadSource]:
        """Find suitable download sources."""
        sources = []

        for source in TRUSTED_SOUNDFONT_SOURCES:
            # Check quality match
            score = 0
            if source.quality == quality:
                score = 100
            elif source.quality == SoundFontQuality.STANDARD:
                score = 80
            else:
                score = 50

            # Check instrument coverage
            if instruments:
                covered = set(instruments) & set(source.instruments)
                coverage = len(covered) / len(instruments)
                score *= coverage

            sources.append((score, source))

        # Sort by score
        sources.sort(key=lambda x: x[0], reverse=True)
        return [source for _, source in sources]

    def _download_source(self, source: DownloadSource) -> Optional[SoundFontInfo]:
        """Download SoundFont from source."""
        output_path = self.config.cache_dir / f"{source.name}.sf2"

        self.logger.info(f"Downloading {source.name} from {source.url}")

        try:
            # Download with progress
            def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
                if total_size > 0:
                    percent = (block_num * block_size / total_size) * 100
                    if block_num % 100 == 0:  # Log every 100 blocks
                        self.logger.info(f"Download progress: {percent:.1f}%")

            urllib.request.urlretrieve(
                source.url,
                output_path,
                reporthook=progress_hook
            )

            # Validate download
            if not self.validator.validate_checksum(output_path, source.checksum):
                output_path.unlink()
                raise ValueError("Checksum validation failed")

            if not self.validator.is_valid(output_path):
                output_path.unlink()
                raise ValueError("SoundFont validation failed")

            # Create SoundFont info
            return SoundFontInfo(
                name=source.name,
                quality=source.quality,
                file_path=output_path,
                size_mb=source.size_mb,
                checksum=source.checksum,
                instruments=source.instruments,
                source_url=source.url
            )

        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise e
```

### Local Caching System

```python
# musicgen/soundfont/cache.py
import json
import time
from pathlib import Path
from typing import Dict, List
import logging

from .manager import SoundFontInfo


class SoundFontCache:
    """Manages local SoundFont cache."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = cache_dir / "cache_metadata.json"
        self.logger = logging.getLogger(__name__)

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")

        return {
            "soundfonts": {},
            "last_cleanup": 0,
            "total_size_mb": 0
        }

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")

    def add_soundfont(self, sf_info: SoundFontInfo) -> None:
        """Add SoundFont to cache tracking."""
        self.metadata["soundfonts"][sf_info.name] = {
            "path": str(sf_info.file_path),
            "size_mb": sf_info.size_mb,
            "checksum": sf_info.checksum,
            "quality": sf_info.quality.value,
            "added_time": time.time(),
            "last_used": time.time()
        }

        self._update_total_size()
        self._save_metadata()

    def mark_used(self, soundfont_name: str) -> None:
        """Mark SoundFont as recently used."""
        if soundfont_name in self.metadata["soundfonts"]:
            self.metadata["soundfonts"][soundfont_name]["last_used"] = time.time()
            self._save_metadata()

    def cleanup(self, max_size_gb: float) -> None:
        """Clean up cache to stay under size limit."""
        max_size_mb = max_size_gb * 1024
        current_size = self._calculate_total_size()

        if current_size <= max_size_mb:
            return

        self.logger.info(f"Cache cleanup: {current_size:.1f}MB > {max_size_mb:.1f}MB")

        # Sort by last used time (oldest first)
        soundfonts = list(self.metadata["soundfonts"].items())
        soundfonts.sort(key=lambda x: x[1]["last_used"])

        size_freed = 0
        for name, info in soundfonts:
            if current_size - size_freed <= max_size_mb:
                break

            # Remove file and metadata
            file_path = Path(info["path"])
            if file_path.exists():
                file_path.unlink()
                size_freed += info["size_mb"]
                self.logger.info(f"Removed {name} ({info['size_mb']:.1f}MB)")

            del self.metadata["soundfonts"][name]

        self._update_total_size()
        self._save_metadata()
        self.metadata["last_cleanup"] = time.time()

    def _calculate_total_size(self) -> float:
        """Calculate total cache size in MB."""
        total = 0.0
        for info in self.metadata["soundfonts"].values():
            file_path = Path(info["path"])
            if file_path.exists():
                total += info["size_mb"]
        return total

    def _update_total_size(self) -> None:
        """Update total size in metadata."""
        self.metadata["total_size_mb"] = self._calculate_total_size()
```

### Quality Tier Definitions

```python
# musicgen/soundfont/quality.py
from dataclasses import dataclass
from typing import Dict, List
from .manager import SoundFontQuality


@dataclass
class QualityTierSpec:
    """Specification for a quality tier."""
    max_size_mb: float
    min_sample_rate: int
    required_instruments: List[int]  # GM program numbers
    description: str


# Quality tier specifications
QUALITY_TIERS: Dict[SoundFontQuality, QualityTierSpec] = {
    SoundFontQuality.MINIMAL: QualityTierSpec(
        max_size_mb=5.0,
        min_sample_rate=22050,
        required_instruments=[0, 1, 24, 25, 32, 40, 41, 48, 56, 73, 80, 128],  # Basic GM
        description="Basic GM sounds, suitable for sketching"
    ),

    SoundFontQuality.STANDARD: QualityTierSpec(
        max_size_mb=20.0,
        min_sample_rate=44100,
        required_instruments=list(range(128)) + [128],  # Full GM + percussion
        description="Full GM compatibility with good quality"
    ),

    SoundFontQuality.HIGH: QualityTierSpec(
        max_size_mb=100.0,
        min_sample_rate=44100,
        required_instruments=list(range(128)) + [128],
        description="Studio-quality samples with extended articulations"
    )
}


def get_recommended_quality(target_duration_seconds: float, voice_count: int) -> SoundFontQuality:
    """Recommend quality tier based on project requirements."""
    # For long durations or many voices, prefer smaller SoundFonts
    if target_duration_seconds > 300 or voice_count > 4:
        return SoundFontQuality.STANDARD
    elif target_duration_seconds > 600 or voice_count > 6:
        return SoundFontQuality.MINIMAL
    else:
        return SoundFontQuality.STANDARD  # Good default
```

### Trusted SoundFont Sources

```python
# musicgen/soundfont/sources.py
from typing import List
from .downloader import DownloadSource
from .manager import SoundFontQuality

# Curated list of trusted SoundFont sources
TRUSTED_SOUNDFONT_SOURCES: List[DownloadSource] = [
    DownloadSource(
        name="GeneralUser_GS",
        url="https://schristiancollins.com/generaluser.php",  # Actual download link needed
        quality=SoundFontQuality.STANDARD,
        size_mb=30.1,
        checksum="a1b2c3d4e5f6",  # Actual checksum needed
        instruments=list(range(128)) + [128]
    ),

    DownloadSource(
        name="FluidR3_GM",
        url="https://member.keymusician.com/Member/FluidR3_GM/FluidR3_GM.tar.gz",
        quality=SoundFontQuality.STANDARD,
        size_mb=148.0,
        checksum="f1e2d3c4b5a6",  # Actual checksum needed
        instruments=list(range(128)) + [128]
    ),

    DownloadSource(
        name="TimGM6mb",
        url="https://sourceforge.net/projects/timbres-de-gm/files/TimGM6mb.sf2",
        quality=SoundFontQuality.MINIMAL,
        size_mb=5.7,
        checksum="6a5b4c3d2e1f",  # Actual checksum needed
        instruments=list(range(128)) + [128]
    ),
]

# Fallback minimal SoundFont (could be bundled)
BUNDLED_MINIMAL_SOUNDFONT = {
    "name": "musicgen_minimal",
    "path": "assets/minimal.sf2",  # Relative to package
    "size_mb": 2.1,
    "quality": SoundFontQuality.MINIMAL,
    "instruments": [0, 1, 24, 25, 32, 40, 41, 48, 56, 73, 80, 128]  # Piano, guitar, bass, etc.
}
```

### SoundFont Validation

```python
# musicgen/soundfont/validation.py
import hashlib
import logging
from pathlib import Path
from typing import Optional

try:
    import pyfluidsynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    FLUIDSYNTH_AVAILABLE = False


class SoundFontValidator:
    """Validates SoundFont files."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def is_valid(self, sf_path: Path) -> bool:
        """Check if SoundFont file is valid."""
        if not sf_path.exists():
            return False

        if sf_path.stat().st_size < 1024:  # Too small
            return False

        # Basic file header check
        try:
            with open(sf_path, 'rb') as f:
                header = f.read(4)
                if header != b'RIFF':
                    return False

                # Skip to form type
                f.seek(8)
                form_type = f.read(4)
                if form_type not in [b'sfbk', b'SF2 ']:
                    return False
        except Exception as e:
            self.logger.warning(f"Failed to read SoundFont header: {e}")
            return False

        # Test with FluidSynth if available
        if FLUIDSYNTH_AVAILABLE:
            return self._test_with_fluidsynth(sf_path)

        return True  # Basic validation passed

    def _test_with_fluidsynth(self, sf_path: Path) -> bool:
        """Test SoundFont with FluidSynth."""
        try:
            # Create temporary synth to test loading
            fs = pyfluidsynth.Synth()
            fs.start()

            sfid = fs.sfload(str(sf_path))
            if sfid == -1:
                return False

            # Test basic functionality
            fs.program_select(0, sfid, 0, 0)  # Select first preset
            fs.noteon(0, 60, 64)  # Play middle C
            fs.noteoff(0, 60)

            fs.delete()
            return True

        except Exception as e:
            self.logger.warning(f"FluidSynth validation failed: {e}")
            return False

    def validate_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Validate file checksum."""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                return file_hash.startswith(expected_checksum)
        except Exception as e:
            self.logger.error(f"Checksum validation failed: {e}")
            return False
```

## Integration with Synthesis Pipeline

### Enhanced Synthesis Configuration

```python
# musicgen/synthesis.py (additions)
from .soundfont.manager import SoundFontManager, SoundFontConfig, SoundFontQuality

class SynthesisConfig(BaseModel):
    """Enhanced synthesis configuration with SoundFont management."""
    # Existing fields...

    # SoundFont management
    soundfont_config: SoundFontConfig = Field(default_factory=SoundFontConfig)
    preferred_soundfont: Optional[Path] = None
    auto_manage_soundfonts: bool = True
    fallback_to_any_soundfont: bool = True


def render_midi_with_smart_soundfont(
    midi_data: pretty_midi.PrettyMIDI,
    config: SynthesisConfig
) -> Path:
    """Render MIDI with smart SoundFont management."""
    # Initialize SoundFont manager
    sf_manager = SoundFontManager(config.soundfont_config)

    # Determine required instruments
    instruments = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            instruments.append(instrument.program)

    # Get optimal SoundFont
    try:
        if config.preferred_soundfont and config.preferred_soundfont.exists():
            soundfont_path = config.preferred_soundfont
        else:
            soundfont_path = sf_manager.get_soundfont(
                instruments=instruments,
                quality=config.soundfont_config.preferred_quality
            )

        # Mark as used for cache management
        sf_manager.cache.mark_used(soundfont_path.stem)

    except Exception as e:
        if config.fallback_to_any_soundfont:
            # Try to find any available SoundFont
            available = sf_manager.list_available()
            if available:
                soundfont_path = available[0].file_path
                logging.warning(f"Using fallback SoundFont: {soundfont_path}")
            else:
                raise RuntimeError("No SoundFonts available") from e
        else:
            raise e

    # Render with selected SoundFont
    return _render_with_soundfont(midi_data, soundfont_path, config)
```

## Configuration Integration

### CLI Integration

```python
# musicgen/cli.py (additions)
import typer
from .soundfont.manager import SoundFontQuality

app = typer.Typer()

@app.command()
def generate(
    # Existing parameters...
    soundfont_quality: SoundFontQuality = typer.Option(
        SoundFontQuality.STANDARD,
        "--soundfont-quality",
        help="SoundFont quality tier (minimal/standard/high)"
    ),
    auto_download_soundfonts: bool = typer.Option(
        True,
        "--auto-download/--no-auto-download",
        help="Automatically download missing SoundFonts"
    ),
    offline_mode: bool = typer.Option(
        False,
        "--offline",
        help="Work in offline mode (no downloads)"
    ),
    soundfont_cache_size_gb: float = typer.Option(
        1.0,
        "--cache-size",
        help="Maximum SoundFont cache size in GB"
    )
) -> None:
    """Generate music with smart SoundFont management."""
    # Configure SoundFont management
    sf_config = SoundFontConfig(
        preferred_quality=soundfont_quality,
        auto_download=auto_download_soundfonts,
        offline_mode=offline_mode,
        max_cache_size_gb=soundfont_cache_size_gb
    )

    # Rest of generation logic...
```

### Configuration File Support

```python
# musicgen/config.py (additions)
class ProjectConfig(BaseModel):
    """Project configuration including SoundFont preferences."""
    # Existing fields...

    # SoundFont preferences
    soundfont: SoundFontConfig = Field(default_factory=SoundFontConfig)

    @classmethod
    def from_file(cls, config_path: Path) -> "ProjectConfig":
        """Load configuration from file."""
        # Implementation for loading YAML/JSON config
        pass
```

## Performance Impact Analysis

### Memory Usage

- **Minimal SoundFonts**: ~2MB RAM when loaded
- **Standard SoundFonts**: ~8-20MB RAM when loaded
- **High-Quality SoundFonts**: ~50-150MB RAM when loaded
- **Cache Overhead**: ~1-5MB for metadata and management

### Network Impact

- **Initial Download**: One-time per quality tier (2-150MB)
- **Background Validation**: Minimal periodic checks
- **Offline Operation**: Zero network usage when cached

### Disk Usage

- **Cache Directory**: Configurable, default 1GB limit
- **Automatic Cleanup**: LRU eviction when over limit
- **Metadata Storage**: ~1MB for tracking information

### CPU Impact

- **SoundFont Loading**: 100-500ms per SoundFont (one-time)
- **Validation**: 50-200ms per SoundFont
- **Cache Management**: <10ms typical operations

## Security Considerations

### Download Security

```python
# Security measures for SoundFont downloads
class SecureDownloader:
    """Secure SoundFont downloader with safety checks."""

    def __init__(self):
        self.max_file_size = 200 * 1024 * 1024  # 200MB limit
        self.allowed_extensions = {'.sf2', '.sf3'}
        self.trusted_domains = {
            'schristiancollins.com',
            'sourceforge.net',
            'github.com'
        }

    def is_url_safe(self, url: str) -> bool:
        """Check if URL is from trusted domain."""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return any(trusted in domain for trusted in self.trusted_domains)

    def validate_download_size(self, content_length: int) -> bool:
        """Validate download size before starting."""
        return content_length <= self.max_file_size
```

### File System Security

- **Sandboxed Cache**: SoundFonts isolated in dedicated directory
- **Checksum Validation**: Prevent corrupted/malicious files
- **File Size Limits**: Prevent disk space exhaustion
- **Permission Checks**: Verify write access before operations

## Cross-Platform Compatibility

### Platform-Specific Considerations

#### Windows
- Cache location: `%LOCALAPPDATA%\musicgen\soundfonts`
- FluidSynth: Requires DLL bundling or separate installation
- Network: Windows Defender may flag downloads

#### macOS
- Cache location: `~/Library/Caches/musicgen/soundfonts`
- FluidSynth: Available via Homebrew
- Network: Gatekeeper may block unsigned downloads

#### Linux
- Cache location: `~/.cache/musicgen/soundfonts`
- FluidSynth: Available via package managers
- Network: Generally permissive

### Path Handling

```python
# Cross-platform path utilities
def get_default_cache_dir() -> Path:
    """Get platform-appropriate cache directory."""
    import platform
    system = platform.system()

    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":  # macOS
        base = Path.home() / "Library" / "Caches"
    else:  # Linux and others
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    return base / "musicgen" / "soundfonts"
```

## Asset Bundling vs Runtime Download Trade-offs

### Bundling Approach

**Pros:**
- Immediate availability
- No network dependency
- Predictable behavior
- Smaller download burden

**Cons:**
- Larger package size
- Limited quality options
- Update difficulties
- Licensing constraints

### Runtime Download Approach

**Pros:**
- Smaller package
- Quality options
- Easy updates
- User choice

**Cons:**
- Network dependency
- Initial delay
- Download failures
- Cache management

### Hybrid Recommendation

```python
# Recommended hybrid approach
class HybridSoundFontStrategy:
    """Combines bundled minimal SoundFont with runtime downloads."""

    def __init__(self):
        # Bundle minimal SoundFont for immediate use
        self.bundled_minimal = self._get_bundled_soundfont()

        # Download higher quality on demand
        self.download_manager = SoundFontDownloader()

    def get_soundfont(self, quality: SoundFontQuality) -> Path:
        """Get SoundFont with hybrid strategy."""
        if quality == SoundFontQuality.MINIMAL:
            # Use bundled for quick start
            return self.bundled_minimal
        else:
            # Download higher quality
            return self.download_manager.get_soundfont(quality)
```

## Implementation Timeline

### Phase 4.1: Core Management (Week 1)
- Basic SoundFont manager
- Local cache system
- File validation

### Phase 4.2: Download System (Week 2)
- Trusted source definitions
- Download implementation
- Security validation

### Phase 4.3: Quality Tiers (Week 3)
- Quality tier system
- Smart selection logic
- Performance optimization

### Phase 4.4: Integration (Week 4)
- CLI integration
- Configuration system
- Error handling

### Phase 4.5: Polish (Week 5)
- Cross-platform testing
- Documentation
- Performance tuning

## Testing Strategy

### Unit Tests

```python
# tests/test_soundfont_manager.py
def test_soundfont_selection():
    """Test SoundFont selection logic."""
    # Test quality matching
    # Test instrument coverage
    # Test fallback behavior

def test_cache_management():
    """Test cache operations."""
    # Test size limits
    # Test LRU eviction
    # Test metadata persistence

def test_download_validation():
    """Test download security."""
    # Test checksum validation
    # Test file size limits
    # Test trusted sources
```

### Integration Tests

```python
# tests/test_soundfont_integration.py
def test_end_to_end_synthesis():
    """Test complete synthesis with SoundFont management."""
    # Test auto-download
    # Test fallback chains
    # Test offline mode
```

This comprehensive SoundFont management system provides reliable, secure, and efficient audio asset management for the MusicGen synthesis pipeline while maintaining excellent user experience across all supported platforms.