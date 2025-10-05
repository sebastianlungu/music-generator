"""
Orchestration module for MusicGen.

This module coordinates the complete pipeline:
1. Load and analyze MIDI files
2. Generate arrangements
3. Synthesize audio
4. Write output files
"""

import logging
from pathlib import Path

from . import analysis, arrange, io_files, synthesis
from .audio_types import (
    AudioCapability,
    get_installation_mode,
    get_missing_dependencies,
    is_capability_available,
    validate_export_formats,
)
from .config import AnalysisResult, ArrangementConfig, MusicGenConfig

# Set up logging
logger = logging.getLogger(__name__)


class GenerationResult:
    """Result of a complete music generation process."""

    def __init__(
        self,
        config: MusicGenConfig,
        analysis_result: AnalysisResult,
        output_files: dict[str, Path],
        success: bool = True,
        error_message: str = "",
    ):
        self.config = config
        self.analysis_result = analysis_result
        self.output_files = output_files
        self.success = success
        self.error_message = error_message

    def __str__(self) -> str:
        if self.success:
            return f"Generation successful. Files written to: {list(self.output_files.values())}"
        else:
            return f"Generation failed: {self.error_message}"


def generate_arrangement(
    config: MusicGenConfig, arrangement_config: ArrangementConfig | None = None
) -> GenerationResult:
    """
    Generate a complete musical arrangement from configuration.

    This is the main entry point for the MusicGen pipeline.

    Args:
        config: Main configuration
        arrangement_config: Optional arrangement-specific configuration

    Returns:
        GenerationResult containing results and output file paths

    Raises:
        DependencyError: If required audio capabilities are missing
    """
    logger.info(f"Starting generation with input: {config.input_path}")

    # Validate export formats upfront - fail fast if dependencies missing
    export_format_strs = [fmt.value for fmt in config.export_formats]
    validated_formats = validate_export_formats(export_format_strs)
    logger.debug(f"Validated export formats: {validated_formats}")

    # Log current installation mode
    installation_mode = get_installation_mode()
    logger.info(f"Installation mode: {installation_mode}")

    try:
        # Discover MIDI files
        midi_files = io_files.discover_midi_files(config.input_path)
        logger.info(f"Found {len(midi_files)} MIDI file(s)")

        # Process each MIDI file (for now, just process the first one)
        # TODO: Add support for processing multiple files or combining them
        input_file = midi_files[0]
        logger.info(f"Processing: {input_file}")

        # Load MIDI file
        midi_data = io_files.load_midi_file(input_file)
        logger.info("MIDI file loaded successfully")

        # Analyze MIDI file
        analysis_result = analysis.analyze_midi_file(midi_data)
        logger.info(
            f"Analysis complete: Key={analysis_result.key}, "
            f"Tempo={analysis_result.tempo:.1f}, "
            f"Duration={analysis_result.duration_seconds:.1f}s"
        )

        # Generate arrangement
        logger.info("Generating arrangement...")
        generated_midi = arrange.generate_arrangement(
            config, analysis_result, arrangement_config
        )
        logger.info("Arrangement generated successfully")

        # Create output directory
        output_dir = io_files.create_output_directory(config.output_dir, input_file)
        logger.info(f"Output directory: {output_dir}")

        # Generate rationale text
        rationale = _generate_rationale(config, analysis_result)

        # Write MIDI file first (needed for mido-based synthesis)
        output_files = io_files.write_output_files(
            output_dir, config, analysis_result, generated_midi, None, rationale
        )

        # Synthesize audio if required by export formats
        audio_data = None
        needs_audio = any(fmt in ["wav", "mp3"] for fmt in validated_formats)

        if needs_audio:
            logger.info("Audio synthesis required for requested export formats")

            # Use mido-based synthesis with the saved MIDI file
            midi_file_path = output_files.get("midi")
            if midi_file_path and midi_file_path.exists():
                try:
                    audio_data = synthesis.synthesize_midi_with_mido(
                        midi_file_path, config.sample_rate
                    )
                    logger.info("Mido-based audio synthesis complete")
                except Exception as e:
                    logger.error(f"Audio synthesis failed: {e}")
                    raise synthesis.SynthesisError(
                        f"Audio synthesis failed: {e}"
                    ) from e
            else:
                raise synthesis.SynthesisError("MIDI file not available for synthesis")
        else:
            logger.info("No audio synthesis needed (MIDI-only export)")

        # Update output files with audio data if generated
        if audio_data is not None:
            output_files = io_files.write_output_files(
                output_dir,
                config,
                analysis_result,
                generated_midi,
                audio_data,
                rationale,
            )

        logger.info(f"Generation complete. Files written: {list(output_files.keys())}")

        return GenerationResult(
            config=config,
            analysis_result=analysis_result,
            output_files=output_files,
            success=True,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return GenerationResult(
            config=config,
            analysis_result=AnalysisResult(
                key="C major",
                tempo=120.0,
                time_signature=(4, 4),
                duration_seconds=0.0,
                pitch_histogram=[0.0] * 12,
                note_density=0.0,
                sections=[],
                instrument_programs=[],
            ),
            output_files={},
            success=False,
            error_message=str(e),
        )


def batch_generate(
    input_directory: Path, output_directory: Path, config_template: MusicGenConfig
) -> list[GenerationResult]:
    """
    Generate arrangements for multiple MIDI files in batch.

    Args:
        input_directory: Directory containing MIDI files
        output_directory: Base output directory
        config_template: Template configuration (input_path will be overridden)

    Returns:
        List of GenerationResult objects
    """
    logger.info(f"Starting batch generation from: {input_directory}")

    try:
        midi_files = io_files.discover_midi_files(input_directory)
    except ValueError as e:
        logger.error(f"No MIDI files found: {e}")
        return []

    results = []

    for i, midi_file in enumerate(midi_files):
        logger.info(f"Processing file {i + 1}/{len(midi_files)}: {midi_file}")

        # Create config for this file
        file_config = config_template.copy(deep=True)
        file_config.input_path = midi_file
        file_config.output_dir = output_directory

        # Generate arrangement
        result = generate_arrangement(file_config)
        results.append(result)

        if result.success:
            logger.info(f"✓ Successfully processed: {midi_file}")
        else:
            logger.error(f"✗ Failed to process: {midi_file} - {result.error_message}")

    successful = sum(1 for r in results if r.success)
    logger.info(f"Batch generation complete: {successful}/{len(results)} successful")

    return results


def _generate_rationale(config: MusicGenConfig, analysis_result: AnalysisResult) -> str:
    """
    Generate a human-readable rationale for the arrangement choices.

    Args:
        config: Configuration used
        analysis_result: Analysis results

    Returns:
        Rationale text
    """
    rationale_parts = []

    # Key and tempo decisions
    if config.key:
        rationale_parts.append(f"Key set to {config.key.value} as specified.")
    else:
        rationale_parts.append(
            f"Key auto-detected as {analysis_result.key} from input analysis."
        )

    if config.tempo_bpm:
        rationale_parts.append(f"Tempo set to {config.tempo_bpm} BPM as specified.")
    elif config.tempo_range:
        min_tempo, max_tempo = config.tempo_range
        rationale_parts.append(
            f"Tempo constrained to {min_tempo}-{max_tempo} BPM range. "
            f"Original tempo was {analysis_result.tempo:.1f} BPM."
        )
    else:
        rationale_parts.append(
            f"Tempo maintained at {analysis_result.tempo:.1f} BPM from input."
        )

    # Instrumentation choices
    instrument_names = [inst.value for inst in config.instruments]
    rationale_parts.append(
        f"Arrangement created for {config.voices} voice(s) using: "
        f"{', '.join(instrument_names)}."
    )

    # Style considerations
    if config.style:
        rationale_parts.append(
            f"Musical style '{config.style}' influenced harmonic progressions "
            f"and rhythmic patterns."
        )

    # Duration handling
    if config.duration_seconds != analysis_result.duration_seconds:
        rationale_parts.append(
            f"Duration adjusted from {analysis_result.duration_seconds:.1f}s "
            f"to {config.duration_seconds}s as specified."
        )

    # Note density considerations
    if analysis_result.note_density > 0:
        density_description = (
            "high"
            if analysis_result.note_density > 5
            else "moderate"
            if analysis_result.note_density > 2
            else "low"
        )
        rationale_parts.append(
            f"Original note density was {density_description} "
            f"({analysis_result.note_density:.1f} notes/second), "
            f"which influenced the generated rhythmic complexity."
        )

    return " ".join(rationale_parts)


def _log_audio_capabilities_summary(logger: logging.Logger) -> None:
    """
    Log a summary of available audio capabilities.

    Args:
        logger: Logger instance
    """
    synthesis_available = is_capability_available(AudioCapability.AUDIO_SYNTHESIS)
    conversion_available = is_capability_available(AudioCapability.AUDIO_CONVERSION)
    full_audio_available = is_capability_available(AudioCapability.FULL_AUDIO)

    if full_audio_available:
        logger.debug("Audio capabilities: Full audio pipeline available")
    else:
        capabilities = []
        if synthesis_available:
            capabilities.append("synthesis")
        if conversion_available:
            capabilities.append("conversion")

        if capabilities:
            logger.debug(f"Audio capabilities: {', '.join(capabilities)} available")
        else:
            logger.debug("Audio capabilities: MIDI-only mode")


def get_audio_status() -> dict[str, bool]:
    """
    Get the current status of audio capabilities.

    Returns:
        Dictionary with capability status
    """
    return {
        "midi_only": is_capability_available(AudioCapability.MIDI_ONLY),
        "audio_synthesis": is_capability_available(AudioCapability.AUDIO_SYNTHESIS),
        "audio_conversion": is_capability_available(AudioCapability.AUDIO_CONVERSION),
        "full_audio": is_capability_available(AudioCapability.FULL_AUDIO),
    }


def suggest_audio_setup_improvements() -> list[str]:
    """
    Suggest improvements for audio setup.

    Returns:
        List of suggestion strings
    """
    suggestions = []

    if not is_capability_available(AudioCapability.AUDIO_SYNTHESIS):
        missing_deps = get_missing_dependencies(AudioCapability.AUDIO_SYNTHESIS)
        suggestions.append(f"For audio synthesis: Install {', '.join(missing_deps)}")

    if not is_capability_available(AudioCapability.AUDIO_CONVERSION):
        missing_deps = get_missing_dependencies(AudioCapability.AUDIO_CONVERSION)
        suggestions.append(
            f"For audio conversion (MP3): Install {', '.join(missing_deps)}"
        )

    return suggestions


def analyze_only(input_path: Path) -> AnalysisResult:
    """
    Perform only analysis without generation.

    Args:
        input_path: Path to MIDI file or directory

    Returns:
        AnalysisResult

    Raises:
        ValueError: If analysis fails
    """
    logger.info(f"Analyzing: {input_path}")

    midi_files = io_files.discover_midi_files(input_path)
    if not midi_files:
        raise ValueError(f"No MIDI files found in {input_path}")

    # For now, analyze the first file
    # TODO: Support for analyzing multiple files and combining results
    midi_file = midi_files[0]
    midi_data = io_files.load_midi_file(midi_file)

    analysis_result = analysis.analyze_midi_file(midi_data)
    logger.info(f"Analysis complete for {midi_file}")

    return analysis_result


def validate_configuration(config: MusicGenConfig) -> list[str]:
    """
    Validate configuration and return list of issues.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Check input path
    if not config.input_path.exists():
        issues.append(f"Input path does not exist: {config.input_path}")

    # Check for MIDI files
    try:
        midi_files = io_files.discover_midi_files(config.input_path)
        if not midi_files:
            issues.append("No MIDI files found in input path")
    except Exception as e:
        issues.append(f"Cannot access input path: {e}")

    # Check SoundFont if audio synthesis is requested
    if config.soundfont_path:
        if not io_files.validate_soundfont(config.soundfont_path):
            issues.append(f"Invalid SoundFont file: {config.soundfont_path}")

    # Check output directory
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create output directory: {e}")

    # Check audio synthesis capability if SoundFont specified
    if config.soundfont_path:
        if not is_capability_available(AudioCapability.AUDIO_SYNTHESIS):
            missing_deps = get_missing_dependencies(AudioCapability.AUDIO_SYNTHESIS)
            issues.append(
                f"Audio synthesis not available but SoundFont specified. "
                f"Missing dependencies: {', '.join(missing_deps)}"
            )

        # Check audio conversion capability if MP3 export requested
        from .config import ExportFormat

        if ExportFormat.MP3 in config.export_formats:
            if not is_capability_available(AudioCapability.AUDIO_CONVERSION):
                missing_deps = get_missing_dependencies(
                    AudioCapability.AUDIO_CONVERSION
                )
                issues.append(
                    f"MP3 export requested but audio conversion not available. "
                    f"Missing dependencies: {', '.join(missing_deps)}"
                )

    return issues


def get_generation_summary(result: GenerationResult) -> dict:
    """
    Get a summary of generation results suitable for display.

    Args:
        result: GenerationResult to summarize

    Returns:
        Dictionary with summary information
    """
    if not result.success:
        return {"success": False, "error": result.error_message, "files_generated": 0}

    analysis = result.analysis_result
    files_info = {}

    for file_type, file_path in result.output_files.items():
        files_info[file_type] = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        }

    return {
        "success": True,
        "input_analysis": {
            "key": analysis.key,
            "tempo": round(analysis.tempo, 1),
            "time_signature": f"{analysis.time_signature[0]}/{analysis.time_signature[1]}",
            "duration": round(analysis.duration_seconds, 1),
            "note_density": round(analysis.note_density, 2),
            "instruments": len(analysis.instrument_programs),
        },
        "generation_config": {
            "target_duration": result.config.duration_seconds,
            "instruments": [inst.value for inst in result.config.instruments],
            "voices": result.config.voices,
            "style": result.config.style,
            "export_formats": [fmt.value for fmt in result.config.export_formats],
        },
        "files_generated": len(result.output_files),
        "files": files_info,
        "audio_capabilities": get_audio_status(),
    }
