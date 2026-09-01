"""Validated instrument-type knowledge catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "metadata" / "instrument-types"
)
SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_KNOWLEDGE_STATUSES = frozenset({"confirmed", "provisional"})
SUPPORTED_NORMALIZATION_STRATEGIES = frozenset({"nearest_allowed_integer"})


class MetadataValidationError(ValueError):
    """Raised when the instrument-type catalog violates its interface."""


@dataclass(frozen=True)
class ScaleDefinition:
    minimum: float
    maximum: float
    minor_division: float


@dataclass(frozen=True)
class ReadingNormalization:
    strategy: str
    maximum_distance: float


@dataclass(frozen=True)
class DialArcDefinition:
    start_angle_degrees: float
    end_angle_degrees: float
    direction: str
    endpoint_snap_degrees: float
    quantize_to_minor_division: bool


@dataclass(frozen=True)
class ReadoutChannel:
    channel_id: str
    name_zh: str
    display_type: str
    quantity_kind: str
    unit: str
    value_format: str
    semantics_zh: str
    scales: tuple[ScaleDefinition, ...]
    allowed_values: tuple[str, ...]
    reading_normalization: ReadingNormalization | None
    dial_arc: DialArcDefinition | None


@dataclass(frozen=True)
class KnowledgeSource:
    source_type: str
    title: str
    url: str | None
    reference: str | None
    retrieved_at: str | None
    local_path: Path | None
    sha256: str | None
    document_code: str | None
    language: str | None


@dataclass(frozen=True)
class InstrumentTypeMetadata:
    type_id: str
    canonical_name_zh: str
    aliases_zh: tuple[str, ...]
    model_markings: tuple[str, ...]
    recognition_signatures: tuple[tuple[str, ...], ...]
    knowledge_status: str
    summary_zh: str
    readout_channels: tuple[ReadoutChannel, ...]
    interpretation_rules_zh: tuple[str, ...]
    sources: tuple[KnowledgeSource, ...]

    def channel(self, channel_id: str) -> ReadoutChannel:
        for channel in self.readout_channels:
            if channel.channel_id == channel_id:
                return channel
        raise KeyError(f"Unknown readout channel for {self.type_id}: {channel_id}")


class InstrumentMetadataCatalog:
    """Load, validate, and query instrument-type knowledge through one seam."""

    def __init__(self, instrument_types: tuple[InstrumentTypeMetadata, ...]):
        self.instrument_types = instrument_types
        self._by_id = {item.type_id: item for item in instrument_types}
        if len(self._by_id) != len(instrument_types):
            raise MetadataValidationError("Instrument type IDs must be unique")

    @classmethod
    def load(cls, path: Path = DEFAULT_CATALOG_PATH) -> InstrumentMetadataCatalog:
        if path.is_dir():
            metadata_files = sorted(path.glob("*/metadata.json"))
            if not metadata_files:
                raise MetadataValidationError(
                    f"No instrument metadata files found under: {path}"
                )
        else:
            metadata_files = [path]

        instrument_types: list[InstrumentTypeMetadata] = []
        for metadata_file in metadata_files:
            instrument_types.extend(_load_metadata_file(metadata_file))
        return cls(tuple(instrument_types))

    def get(self, type_id: str) -> InstrumentTypeMetadata:
        try:
            return self._by_id[type_id]
        except KeyError as error:
            raise KeyError(f"Unknown instrument type: {type_id}") from error

    def find(self, visible_text: str) -> tuple[InstrumentTypeMetadata, ...]:
        query = _normalize(visible_text)
        if not query:
            return ()
        matches = []
        for item in self.instrument_types:
            terms = (
                item.canonical_name_zh,
                *item.aliases_zh,
                *item.model_markings,
            )
            normalized_terms = tuple(_normalize(term) for term in terms)
            term_match = any(
                term in query or query in term for term in normalized_terms
            )
            signature_match = any(
                all(_normalize(term) in query for term in signature)
                for signature in item.recognition_signatures
            )
            if term_match or signature_match:
                matches.append(item)
        return tuple(matches)


def _load_metadata_file(path: Path) -> tuple[InstrumentTypeMetadata, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise MetadataValidationError(
            f"Invalid metadata JSON in {path}: {error}"
        ) from error
    return _parse_catalog(payload, path.parent)


def _parse_catalog(
    payload: Any, source_directory: Path
) -> tuple[InstrumentTypeMetadata, ...]:
    root = _mapping(payload, "catalog")
    schema_version = root.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise MetadataValidationError(
            f"Unsupported schema_version: {schema_version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )

    has_single_type = "instrument_type" in root
    has_type_list = "instrument_types" in root
    if has_single_type == has_type_list:
        raise MetadataValidationError(
            "catalog requires exactly one of instrument_type or instrument_types"
        )
    raw_types = (
        [root["instrument_type"]]
        if has_single_type
        else _list(root["instrument_types"], "instrument_types")
    )
    if not raw_types:
        raise MetadataValidationError("instrument_types must not be empty")

    parsed = tuple(
        _parse_instrument_type(
            item,
            f"instrument_types[{index}]",
            source_directory,
        )
        for index, item in enumerate(raw_types)
    )
    type_ids = [item.type_id for item in parsed]
    if len(set(type_ids)) != len(type_ids):
        raise MetadataValidationError("Instrument type IDs must be unique")
    return parsed


def _parse_instrument_type(
    value: Any, location: str, source_directory: Path
) -> InstrumentTypeMetadata:
    item = _mapping(value, location)
    status = _string(item.get("knowledge_status"), f"{location}.knowledge_status")
    if status not in SUPPORTED_KNOWLEDGE_STATUSES:
        raise MetadataValidationError(
            f"{location}.knowledge_status must be one of "
            f"{sorted(SUPPORTED_KNOWLEDGE_STATUSES)}"
        )
    channels = tuple(
        _parse_channel(channel, f"{location}.readout_channels[{index}]")
        for index, channel in enumerate(
            _list(item.get("readout_channels"), f"{location}.readout_channels")
        )
    )
    if not channels:
        raise MetadataValidationError(f"{location}.readout_channels must not be empty")
    channel_ids = [channel.channel_id for channel in channels]
    if len(set(channel_ids)) != len(channel_ids):
        raise MetadataValidationError(f"{location} readout channel IDs must be unique")
    sources = tuple(
        _parse_source(
            source,
            f"{location}.sources[{index}]",
            source_directory,
        )
        for index, source in enumerate(
            _list(item.get("sources"), f"{location}.sources")
        )
    )
    if not sources:
        raise MetadataValidationError(f"{location}.sources must not be empty")
    return InstrumentTypeMetadata(
        type_id=_string(item.get("type_id"), f"{location}.type_id"),
        canonical_name_zh=_string(
            item.get("canonical_name_zh"), f"{location}.canonical_name_zh"
        ),
        aliases_zh=_string_tuple(item.get("aliases_zh"), f"{location}.aliases_zh"),
        model_markings=_string_tuple(
            item.get("model_markings"), f"{location}.model_markings"
        ),
        recognition_signatures=_string_matrix(
            item.get("recognition_signatures", []),
            f"{location}.recognition_signatures",
        ),
        knowledge_status=status,
        summary_zh=_string(item.get("summary_zh"), f"{location}.summary_zh"),
        readout_channels=channels,
        interpretation_rules_zh=_string_tuple(
            item.get("interpretation_rules_zh"),
            f"{location}.interpretation_rules_zh",
        ),
        sources=sources,
    )


def _parse_channel(value: Any, location: str) -> ReadoutChannel:
    item = _mapping(value, location)
    scales = tuple(
        _parse_scale(scale, f"{location}.scales[{index}]")
        for index, scale in enumerate(_list(item.get("scales"), f"{location}.scales"))
    )
    allowed_values = _string_tuple(
        item.get("allowed_values", []), f"{location}.allowed_values"
    )
    reading_normalization = _parse_reading_normalization(
        item.get("reading_normalization"),
        f"{location}.reading_normalization",
    )
    dial_arc = _parse_dial_arc(item.get("dial_arc"), f"{location}.dial_arc")
    if dial_arc is not None and not scales:
        raise MetadataValidationError(f"{location}.dial_arc requires scales")
    if reading_normalization is not None:
        if not allowed_values:
            raise MetadataValidationError(
                f"{location}.reading_normalization requires allowed_values"
            )
        try:
            normalized_values = tuple(int(value) for value in allowed_values)
        except ValueError as error:
            raise MetadataValidationError(
                f"{location}.allowed_values must be integers when normalization is set"
            ) from error
        if len(set(normalized_values)) != len(normalized_values):
            raise MetadataValidationError(f"{location}.allowed_values must be unique")
    return ReadoutChannel(
        channel_id=_string(item.get("channel_id"), f"{location}.channel_id"),
        name_zh=_string(item.get("name_zh"), f"{location}.name_zh"),
        display_type=_string(item.get("display_type"), f"{location}.display_type"),
        quantity_kind=_string(item.get("quantity_kind"), f"{location}.quantity_kind"),
        unit=_string(item.get("unit"), f"{location}.unit"),
        value_format=_string(item.get("value_format"), f"{location}.value_format"),
        semantics_zh=_string(item.get("semantics_zh"), f"{location}.semantics_zh"),
        scales=scales,
        allowed_values=allowed_values,
        reading_normalization=reading_normalization,
        dial_arc=dial_arc,
    )


def _parse_reading_normalization(
    value: Any, location: str
) -> ReadingNormalization | None:
    if value is None:
        return None
    item = _mapping(value, location)
    strategy = _string(item.get("strategy"), f"{location}.strategy")
    if strategy not in SUPPORTED_NORMALIZATION_STRATEGIES:
        raise MetadataValidationError(
            f"{location}.strategy must be one of "
            f"{sorted(SUPPORTED_NORMALIZATION_STRATEGIES)}"
        )
    maximum_distance = _number(
        item.get("maximum_distance"), f"{location}.maximum_distance"
    )
    if maximum_distance <= 0:
        raise MetadataValidationError(f"{location}.maximum_distance must be positive")
    return ReadingNormalization(strategy, maximum_distance)


def _parse_scale(value: Any, location: str) -> ScaleDefinition:
    item = _mapping(value, location)
    minimum = _number(item.get("minimum"), f"{location}.minimum")
    maximum = _number(item.get("maximum"), f"{location}.maximum")
    minor_division = _number(item.get("minor_division"), f"{location}.minor_division")
    if maximum <= minimum:
        raise MetadataValidationError(f"{location}.maximum must exceed minimum")
    if minor_division <= 0:
        raise MetadataValidationError(f"{location}.minor_division must be positive")
    return ScaleDefinition(minimum, maximum, minor_division)


def _parse_dial_arc(value: Any, location: str) -> DialArcDefinition | None:
    if value is None:
        return None
    item = _mapping(value, location)
    start = _number(item.get("start_angle_degrees"), f"{location}.start_angle_degrees")
    end = _number(item.get("end_angle_degrees"), f"{location}.end_angle_degrees")
    direction = _string(item.get("direction"), f"{location}.direction")
    endpoint_snap = _number(
        item.get("endpoint_snap_degrees"), f"{location}.endpoint_snap_degrees"
    )
    quantize = _boolean(
        item.get("quantize_to_minor_division", False),
        f"{location}.quantize_to_minor_division",
    )
    if not 0 <= start < 360 or not 0 <= end < 360:
        raise MetadataValidationError(f"{location} angles must be in [0, 360)")
    if start == end:
        raise MetadataValidationError(f"{location} start and end must differ")
    if direction not in {"clockwise", "counterclockwise"}:
        raise MetadataValidationError(
            f"{location}.direction must be clockwise or counterclockwise"
        )
    if not 0 < endpoint_snap < 45:
        raise MetadataValidationError(
            f"{location}.endpoint_snap_degrees must be between 0 and 45"
        )
    return DialArcDefinition(start, end, direction, endpoint_snap, quantize)


def _parse_source(value: Any, location: str, source_directory: Path) -> KnowledgeSource:
    item = _mapping(value, location)
    url = _optional_string(item.get("url"), f"{location}.url")
    reference = _optional_string(item.get("reference"), f"{location}.reference")
    raw_local_path = _optional_string(item.get("local_path"), f"{location}.local_path")
    local_path = _resolve_local_path(raw_local_path, source_directory, location)
    sha256 = _optional_string(item.get("sha256"), f"{location}.sha256")
    if url is None and reference is None and local_path is None:
        raise MetadataValidationError(
            f"{location} requires url, reference, or local_path"
        )
    if sha256 is not None:
        if local_path is None:
            raise MetadataValidationError(f"{location}.sha256 requires a local_path")
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256.lower()
        ):
            raise MetadataValidationError(
                f"{location}.sha256 must be a 64-character hexadecimal digest"
            )
        if local_path.exists():
            actual_sha256 = hashlib.sha256(local_path.read_bytes()).hexdigest()
            if actual_sha256 != sha256.lower():
                raise MetadataValidationError(
                    f"{location}.sha256 does not match {local_path}"
                )
    return KnowledgeSource(
        source_type=_string(item.get("source_type"), f"{location}.source_type"),
        title=_string(item.get("title"), f"{location}.title"),
        url=url,
        reference=reference,
        retrieved_at=_optional_string(
            item.get("retrieved_at"), f"{location}.retrieved_at"
        ),
        local_path=local_path,
        sha256=sha256.lower() if sha256 is not None else None,
        document_code=_optional_string(
            item.get("document_code"), f"{location}.document_code"
        ),
        language=_optional_string(item.get("language"), f"{location}.language"),
    )


def _resolve_local_path(
    value: str | None, source_directory: Path, location: str
) -> Path | None:
    if value is None:
        return None
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise MetadataValidationError(
            f"{location}.local_path must stay inside its instrument-type directory"
        )
    return (source_directory / relative_path).resolve()


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MetadataValidationError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise MetadataValidationError(f"{location} must be an array")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetadataValidationError(f"{location} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, location: str) -> str | None:
    if value is None:
        return None
    return _string(value, location)


def _string_tuple(value: Any, location: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{location}[{index}]")
        for index, item in enumerate(_list(value, location))
    )


def _string_matrix(value: Any, location: str) -> tuple[tuple[str, ...], ...]:
    signatures = tuple(
        _string_tuple(item, f"{location}[{index}]")
        for index, item in enumerate(_list(value, location))
    )
    if any(not signature for signature in signatures):
        raise MetadataValidationError(f"{location} signatures must not be empty")
    return signatures


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MetadataValidationError(f"{location} must be numeric")
    return float(value)


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise MetadataValidationError(f"{location} must be boolean")
    return value


def _normalize(value: str) -> str:
    return "".join(value.casefold().split())


def main() -> int:
    parser = argparse.ArgumentParser(description="Query instrument-type knowledge")
    parser.add_argument("visible_text", help="Model marking or visible instrument name")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    args = parser.parse_args()
    matches = InstrumentMetadataCatalog.load(args.catalog).find(args.visible_text)
    if not matches:
        print("No instrument type metadata matched")
        return 1
    for item in matches:
        print(f"{item.type_id}: {item.canonical_name_zh} [{item.knowledge_status}]")
        print(f"  {item.summary_zh}")
        for channel in item.readout_channels:
            print(
                f"  - {channel.channel_id}: {channel.name_zh} / "
                f"{channel.display_type} / {channel.unit}"
            )
        for rule in item.interpretation_rules_zh:
            print(f"  rule: {rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
