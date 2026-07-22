"""
Team Detectors Package
"""

from .blue_team import (
    BlueTeamDetector, RedTeamDetector, TeamDetector,
    HeroPortraitResult,
    create_blue_team_detector, create_red_team_detector,
)

__all__ = [
    "BlueTeamDetector", "RedTeamDetector", "TeamDetector",
    "HeroPortraitResult",
    "create_blue_team_detector", "create_red_team_detector",
]