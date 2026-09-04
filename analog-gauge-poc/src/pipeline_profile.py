"""Named gauge-processing profiles for reproducible 448/640 comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GaugePipelineProfile:
    """Keep dial-canvas and model-input dimensions independently auditable."""

    name: str
    detection_size: int
    dial_canvas_size: int
    segmentation_inference_size: int
    segmentation_confidence: float
    use_high_resolution_detail: bool
    segment_on_high_resolution_detail: bool
    preserve_canvas_aspect_ratio: bool
    geometry_crop_margin_fraction: float


_PROFILES = {
    "448": GaugePipelineProfile(
        name="448",
        detection_size=640,
        dial_canvas_size=448,
        segmentation_inference_size=448,
        segmentation_confidence=0.12,
        use_high_resolution_detail=False,
        segment_on_high_resolution_detail=False,
        preserve_canvas_aspect_ratio=False,
        geometry_crop_margin_fraction=0.0,
    ),
    "448-model640": GaugePipelineProfile(
        name="448-model640",
        detection_size=640,
        dial_canvas_size=448,
        segmentation_inference_size=640,
        segmentation_confidence=0.12,
        use_high_resolution_detail=False,
        segment_on_high_resolution_detail=False,
        preserve_canvas_aspect_ratio=False,
        geometry_crop_margin_fraction=0.0,
    ),
    "640": GaugePipelineProfile(
        name="640",
        detection_size=640,
        dial_canvas_size=640,
        segmentation_inference_size=640,
        segmentation_confidence=0.12,
        use_high_resolution_detail=False,
        segment_on_high_resolution_detail=False,
        preserve_canvas_aspect_ratio=False,
        geometry_crop_margin_fraction=0.0,
    ),
    "448-highres": GaugePipelineProfile(
        name="448-highres",
        detection_size=640,
        dial_canvas_size=448,
        segmentation_inference_size=448,
        segmentation_confidence=0.12,
        use_high_resolution_detail=True,
        segment_on_high_resolution_detail=False,
        preserve_canvas_aspect_ratio=False,
        geometry_crop_margin_fraction=0.0,
    ),
    "448-highres-pad": GaugePipelineProfile(
        name="448-highres-pad",
        detection_size=640,
        dial_canvas_size=448,
        segmentation_inference_size=448,
        segmentation_confidence=0.12,
        use_high_resolution_detail=True,
        segment_on_high_resolution_detail=False,
        preserve_canvas_aspect_ratio=True,
        geometry_crop_margin_fraction=0.10,
    ),
    "448-highres-seg": GaugePipelineProfile(
        name="448-highres-seg",
        detection_size=640,
        dial_canvas_size=448,
        segmentation_inference_size=448,
        segmentation_confidence=0.12,
        use_high_resolution_detail=True,
        segment_on_high_resolution_detail=True,
        preserve_canvas_aspect_ratio=False,
        geometry_crop_margin_fraction=0.0,
    ),
}

GAUGE_PIPELINE_PROFILE_NAMES = tuple(_PROFILES)
DEFAULT_GAUGE_PIPELINE_PROFILE_NAME = "448-highres-pad"


def get_gauge_pipeline_profile(name: str) -> GaugePipelineProfile:
    """Resolve a versioned profile name or fail closed."""

    try:
        return _PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown gauge pipeline profile: {name}") from error
