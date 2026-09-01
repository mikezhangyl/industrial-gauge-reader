"""Validated local human confirmations, kept separate from type knowledge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.instrument_metadata import InstrumentMetadataCatalog, MetadataValidationError

DEFAULT_OBSERVATION_PATH = Path(__file__).resolve().parents[1] / "observations"


@dataclass(frozen=True)
class ConfirmedReadout:
    instance_id: str
    channel_id: str
    confirmed_value: float | int | str | None
    confirmed_candidates: tuple[float, ...]
    raw_display: str | None
    unit: str
    confirmation_status: str
    note_zh: str | None


@dataclass(frozen=True)
class InstrumentObservation:
    observation_id: str
    image_sha256: str
    instrument_type_id: str
    readouts: tuple[ConfirmedReadout, ...]
    publish_to_public_repository: bool


class InstrumentObservationCatalog:
    """Load local confirmations and resolve them only by image digest."""

    def __init__(self, observations: tuple[InstrumentObservation, ...]):
        self.observations = observations
        self._by_digest = {item.image_sha256: item for item in observations}
        if len(self._by_digest) != len(observations):
            raise MetadataValidationError("Observation image digests must be unique")

    @classmethod
    def load(
        cls,
        metadata_catalog: InstrumentMetadataCatalog,
        path: Path = DEFAULT_OBSERVATION_PATH,
    ) -> InstrumentObservationCatalog:
        observations = tuple(
            _load_observation(file, metadata_catalog)
            for file in sorted(path.glob("*.json"))
        )
        return cls(observations)

    def for_image(self, image_path: Path) -> InstrumentObservation | None:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        return self._by_digest.get(digest)


def _load_observation(
    path: Path, metadata_catalog: InstrumentMetadataCatalog
) -> InstrumentObservation:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MetadataValidationError(
            f"Invalid observation JSON in {path}: {error}"
        ) from error
    root = _mapping(payload, str(path))
    if root.get("schema_version") != 1:
        raise MetadataValidationError(f"Unsupported observation schema in {path}")
    image_sha256 = _string(root.get("image_sha256"), f"{path}.image_sha256").lower()
    if len(image_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in image_sha256
    ):
        raise MetadataValidationError(f"{path}.image_sha256 is invalid")
    instrument_type_id = _string(
        root.get("instrument_type_id"), f"{path}.instrument_type_id"
    )
    metadata = metadata_catalog.get(instrument_type_id)
    readouts = _parse_readouts(root, path)
    if not readouts:
        raise MetadataValidationError(f"{path} requires at least one confirmed readout")
    seen: set[tuple[str, str]] = set()
    for readout in readouts:
        key = (readout.instance_id, readout.channel_id)
        if key in seen:
            raise MetadataValidationError(
                f"Duplicate observation readout in {path}: {key}"
            )
        seen.add(key)
        channel = metadata.channel(readout.channel_id)
        if channel.unit != readout.unit:
            raise MetadataValidationError(
                f"{path} unit mismatch for {readout.channel_id}: "
                f"{readout.unit} != {channel.unit}"
            )
    privacy = _mapping(root.get("privacy"), f"{path}.privacy")
    publish = _boolean(
        privacy.get("publish_to_public_repository"),
        f"{path}.privacy.publish_to_public_repository",
    )
    return InstrumentObservation(
        observation_id=_string(root.get("observation_id"), f"{path}.observation_id"),
        image_sha256=image_sha256,
        instrument_type_id=instrument_type_id,
        readouts=readouts,
        publish_to_public_repository=publish,
    )


def _parse_readouts(root: dict[str, Any], path: Path) -> tuple[ConfirmedReadout, ...]:
    if "instances" in root:
        readouts: list[ConfirmedReadout] = []
        for instance_index, raw_instance in enumerate(
            _list(root["instances"], f"{path}.instances")
        ):
            instance = _mapping(raw_instance, f"{path}.instances[{instance_index}]")
            instance_id = _string(
                instance.get("instance_id"),
                f"{path}.instances[{instance_index}].instance_id",
            )
            for readout_index, raw_readout in enumerate(
                _list(
                    instance.get("readouts"),
                    f"{path}.instances[{instance_index}].readouts",
                )
            ):
                readouts.append(
                    _parse_readout(
                        raw_readout,
                        instance_id,
                        f"{path}.instances[{instance_index}].readouts[{readout_index}]",
                    )
                )
        return tuple(readouts)
    return tuple(
        _parse_readout(raw_readout, "instance_1", f"{path}.readouts[{index}]")
        for index, raw_readout in enumerate(
            _list(root.get("readouts"), f"{path}.readouts")
        )
    )


def _parse_readout(value: Any, instance_id: str, location: str) -> ConfirmedReadout:
    item = _mapping(value, location)
    confirmed_value = item.get("confirmed_value")
    if confirmed_value is not None and (
        isinstance(confirmed_value, bool)
        or not isinstance(confirmed_value, str | int | float)
    ):
        raise MetadataValidationError(f"{location}.confirmed_value is invalid")
    candidates = tuple(
        _number(candidate, f"{location}.confirmed_candidates[{index}]")
        for index, candidate in enumerate(
            _list(
                item.get("confirmed_candidates", []), f"{location}.confirmed_candidates"
            )
        )
    )
    if confirmed_value is None and not candidates:
        raise MetadataValidationError(
            f"{location} requires confirmed_value or confirmed_candidates"
        )
    return ConfirmedReadout(
        instance_id=instance_id,
        channel_id=_string(item.get("channel_id"), f"{location}.channel_id"),
        confirmed_value=confirmed_value,
        confirmed_candidates=candidates,
        raw_display=_optional_string(
            item.get("raw_display"), f"{location}.raw_display"
        ),
        unit=_string(item.get("unit"), f"{location}.unit"),
        confirmation_status=_string(
            item.get("confirmation_status"), f"{location}.confirmation_status"
        ),
        note_zh=_optional_string(item.get("note_zh"), f"{location}.note_zh"),
    )


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
    return None if value is None else _string(value, location)


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MetadataValidationError(f"{location} must be numeric")
    return float(value)


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise MetadataValidationError(f"{location} must be boolean")
    return value
