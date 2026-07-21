"""Vision core modules — layout, frame reader, cropper, pipeline, state builder."""
from . import layout
from .frame_reader import FrameReader
from .cropper import crop, crop_region, crop_multi, crop_all_of_type
from .pipeline import Pipeline, FrameResult, DetectionResult
from .state_builder import StateBuilder, GameState, HeroState, state_to_flat_dict

__all__ = [
    "layout",
    "FrameReader",
    "crop", "crop_region", "crop_multi", "crop_all_of_type",
    "Pipeline", "FrameResult", "DetectionResult",
    "StateBuilder", "GameState", "HeroState", "state_to_flat_dict",
]
