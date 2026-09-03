"""Apply instrument-type knowledge after generic visual measurement."""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.gauge_reader import GaugeResult
from src.instrument_metadata import (
    DialArcDefinition,
    InstrumentMetadataCatalog,
    InstrumentTypeMetadata,
    MetadataValidationError,
    ReadoutChannel,
    ScaleDefinition,
)

POINTER_DISPLAY_TYPES = frozenset(
    {
        "analog_pointer",
        "analog_pointer_with_counterweight",
        "discrete_pointer_dial",
        "single_pointer_circular_counter",
    }
)


@dataclass(frozen=True)
class InstrumentReadingInterpreter:
    """Turn a raw visual reading into a metadata-governed business reading."""

    catalog: InstrumentMetadataCatalog

    def interpret(self, result: GaugeResult, visible_text: str) -> GaugeResult:
        matches = self.catalog.find(visible_text)
        if len(matches) != 1:
            return result
        metadata = matches[0]
        channel = _pointer_channel(metadata)
        if channel is None:
            return replace(result, instrument_type_id=metadata.type_id)

        interpreted = replace(
            result,
            raw_reading=result.reading,
            unit=channel.unit,
            instrument_type_id=metadata.type_id,
            readout_channel_id=channel.channel_id,
        )
        if (
            result.sweep_fraction is not None
            and result.center_method is not None
            and "hidden-pivot+tick-scale" in result.center_method
        ):
            return _interpret_visual_sweep_fraction(interpreted, channel)
        if channel.dial_arc is not None and result.angle_degrees is not None:
            angle_interpreted = _interpret_dial_angle(interpreted, channel)
            if angle_interpreted.interpretation_method != "metadata:dial_arc_rejected":
                return angle_interpreted
        raw_reading = result.reading
        if raw_reading is None:
            return _interpret_dial_angle(interpreted, channel)
        normalization = channel.reading_normalization
        if normalization is None:
            return replace(
                interpreted,
                interpretation_method="metadata:preserve_raw_reading",
            )
        if normalization.strategy != "nearest_allowed_integer":
            raise MetadataValidationError(
                f"Unsupported normalization strategy: {normalization.strategy}"
            )
        if not isinstance(raw_reading, int | float):
            return replace(
                interpreted,
                reading=None,
                interpretation_method="metadata:normalization_rejected",
                failure_reason="Discrete integer normalization requires a numeric reading",
            )

        candidates = _integer_candidates(channel)
        nearest = min(candidates, key=lambda value: abs(raw_reading - value))
        distance = abs(raw_reading - nearest)
        if distance > normalization.maximum_distance:
            return replace(
                interpreted,
                reading=None,
                interpretation_method="metadata:normalization_rejected",
                failure_reason=(
                    "Raw reading is outside the configured normalization distance: "
                    f"{raw_reading:.3f}"
                ),
            )
        return replace(
            interpreted,
            reading=float(nearest),
            interpretation_method="metadata:nearest_allowed_integer",
        )


def _pointer_channel(metadata: InstrumentTypeMetadata) -> ReadoutChannel | None:
    channels = tuple(
        channel
        for channel in metadata.readout_channels
        if channel.display_type in POINTER_DISPLAY_TYPES
    )
    return channels[0] if len(channels) == 1 else None


def _integer_candidates(channel: ReadoutChannel) -> tuple[int, ...]:
    if not channel.allowed_values:
        raise MetadataValidationError(
            f"{channel.channel_id} normalization requires allowed_values"
        )
    try:
        candidates = tuple(int(value) for value in channel.allowed_values)
    except ValueError as error:
        raise MetadataValidationError(
            f"{channel.channel_id} allowed_values must be integers"
        ) from error
    if len(set(candidates)) != len(candidates):
        raise MetadataValidationError(
            f"{channel.channel_id} allowed_values must be unique"
        )
    return candidates


def _interpret_dial_angle(result: GaugeResult, channel: ReadoutChannel) -> GaugeResult:
    arc = channel.dial_arc
    angle = result.angle_degrees
    if arc is None or angle is None or not channel.scales:
        return result

    fraction = _dial_fraction(angle, arc)
    direction_flipped = False
    if fraction is None:
        flipped_angle = (angle + 180.0) % 360.0
        fraction = _dial_fraction(flipped_angle, arc)
        if fraction is not None:
            angle = flipped_angle
            direction_flipped = True
    if fraction is None:
        return replace(
            result,
            interpretation_method="metadata:dial_arc_rejected",
        )

    interpreted_result = result
    if direction_flipped:
        pointer_tip = result.pointer_tip
        center = result.center
        if pointer_tip is not None and center is not None:
            pointer_tip = (
                2.0 * center[0] - pointer_tip[0],
                2.0 * center[1] - pointer_tip[1],
            )
        interpreted_result = replace(
            result,
            angle_degrees=angle,
            pointer_tip=pointer_tip,
        )

    method_suffix = "+direction_flip" if direction_flipped else ""
    candidates = tuple(
        _scale_value(scale, fraction, arc.quantize_to_minor_division)
        for scale in channel.scales
    )
    if len(candidates) == 1:
        return replace(
            interpreted_result,
            reading=candidates[0],
            reading_candidates=(),
            interpretation_method=f"metadata:dial_arc_scale{method_suffix}",
            failure_reason=None,
        )
    return replace(
        interpreted_result,
        reading=None,
        reading_candidates=candidates,
        interpretation_method=(f"metadata:ambiguous_scale_candidates{method_suffix}"),
        failure_reason="The active scale cannot be selected from the image alone",
    )


def _interpret_visual_sweep_fraction(
    result: GaugeResult,
    channel: ReadoutChannel,
) -> GaugeResult:
    fraction = result.sweep_fraction
    arc = channel.dial_arc
    if fraction is None or arc is None or not channel.scales:
        return result
    if not 0.0 <= fraction <= 1.0:
        return replace(
            result,
            reading=None,
            interpretation_method="metadata:visual_sweep_rejected",
            failure_reason="Visual scale fraction falls outside the dial sweep",
        )
    if arc.direction == "clockwise":
        total_sweep = (arc.end_angle_degrees - arc.start_angle_degrees) % 360.0
    else:
        total_sweep = (arc.start_angle_degrees - arc.end_angle_degrees) % 360.0
    if total_sweep > 0.0:
        endpoint_fraction = min(0.5, arc.endpoint_snap_degrees / total_sweep)
        if fraction <= endpoint_fraction:
            fraction = 0.0
        elif 1.0 - fraction <= endpoint_fraction:
            fraction = 1.0
    candidates = tuple(
        _scale_value(scale, fraction, arc.quantize_to_minor_division)
        for scale in channel.scales
    )
    if len(candidates) == 1:
        return replace(
            result,
            reading=candidates[0],
            reading_candidates=(),
            interpretation_method="metadata:visual_sweep_scale",
            failure_reason=None,
        )
    return replace(
        result,
        reading=None,
        reading_candidates=candidates,
        interpretation_method="metadata:visual_sweep_candidates",
        failure_reason="The active scale cannot be selected from the image alone",
    )


def _dial_fraction(angle: float, arc: DialArcDefinition) -> float | None:
    start_angle = arc.start_angle_degrees
    end_angle = arc.end_angle_degrees
    endpoint_snap = arc.endpoint_snap_degrees
    if arc.direction == "clockwise":
        total_sweep = (end_angle - start_angle) % 360.0
        pointer_sweep = (angle - start_angle) % 360.0
    else:
        total_sweep = (start_angle - end_angle) % 360.0
        pointer_sweep = (start_angle - angle) % 360.0

    start_distance = _angular_distance(angle, start_angle)
    end_distance = _angular_distance(angle, end_angle)
    if start_distance <= endpoint_snap:
        return 0.0
    if end_distance <= endpoint_snap:
        return 1.0
    if pointer_sweep <= total_sweep:
        return pointer_sweep / total_sweep
    return None


def _scale_value(scale: ScaleDefinition, fraction: float, quantize: bool) -> float:
    minimum = scale.minimum
    maximum = scale.maximum
    value = minimum + fraction * (maximum - minimum)
    if quantize:
        division = scale.minor_division
        value = minimum + round((value - minimum) / division) * division
    return float(min(maximum, max(minimum, value)))


def _angular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)
