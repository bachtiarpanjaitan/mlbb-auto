"""
Detector modules — masing-masing bertanggung jawab mendeteksi satu aspek
game state dari region yang sudah di-crop sesuai layout.
"""
from .base import BaseDetector, Detection
from .timer import TimerDetector
from .gold import GoldDetector
from .hp import HPDetector
from .mana import ManaDetector
from .level import LevelDetector
from .hero import HeroDetector
from .items import ItemsDetector
from .minimap import MinimapDetector
from .skills import SkillsDetector
from .towers import TowersDetector
from .objectives import ObjectivesDetector

__all__ = [
    "BaseDetector", "Detection",
    "TimerDetector", "GoldDetector", "HPDetector", "ManaDetector",
    "LevelDetector", "HeroDetector", "ItemsDetector", "MinimapDetector",
    "SkillsDetector", "TowersDetector", "ObjectivesDetector",
]
