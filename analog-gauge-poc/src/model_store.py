"""Download and verify the public pretrained pose models used by the PoC."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ModelArtifact:
    filename: str
    url: str
    sha256: str
    size: int


ARTIFACTS = (
    ModelArtifact(
        filename="corners-best.pt",
        url=(
            "https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/"
            "resolve/main/corners-best.pt?download=true"
        ),
        sha256="a88502e86a40941aec69fe4d48e03c675a9381500fbf4c1ca8e3d1a89db089a9",
        size=37_732_202,
    ),
    ModelArtifact(
        filename="keypoints-best.pt",
        url=(
            "https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/"
            "resolve/main/keypoints-best.pt?download=true"
        ),
        sha256="8b9b6ac6b3e5dd73a4e00af18365b13e0607cb8e4b00cfd9189e005b86103124",
        size=6_409_410,
    ),
    ModelArtifact(
        filename="ethz-gauge-detection.pt",
        url=(
            "https://media.githubusercontent.com/media/ethz-asl/"
            "analog_gauge_reader/main/models/gauge_detection_model.pt"
        ),
        sha256="eb496ed0c007b890fe7d26da69a777076cb8f28f0166f2a5905e971d85d15db8",
        size=6_239_224,
    ),
    ModelArtifact(
        filename="ethz-segmentation.pt",
        url=(
            "https://media.githubusercontent.com/media/ethz-asl/"
            "analog_gauge_reader/main/models/segmentation_model.pt"
        ),
        sha256="28efb6aa2ebd9032b55ed29636c937b6410c869e3775435bde2c2ac167536cbf",
        size=6_768_834,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(artifact: ModelArtifact, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = Request(artifact.url, headers={"User-Agent": "analog-gauge-poc/1.0"})
    try:
        with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size != artifact.size:
            raise RuntimeError(
                f"Unexpected size for {artifact.filename}: "
                f"{temporary.stat().st_size} != {artifact.size}"
            )
        actual_hash = _sha256(temporary)
        if actual_hash != artifact.sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {artifact.filename}: {actual_hash}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_models(models_dir: Path) -> dict[str, Path]:
    """Ensure model files exist and match the recorded upstream SHA-256."""
    models_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for artifact in ARTIFACTS:
        destination = models_dir / artifact.filename
        if (
            not destination.exists()
            or destination.stat().st_size != artifact.size
            or _sha256(destination) != artifact.sha256
        ):
            print(f"Downloading verified model: {artifact.filename}")
            _download(artifact, destination)
        resolved[artifact.filename] = destination
    return resolved
