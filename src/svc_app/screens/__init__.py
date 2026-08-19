"""Application screens."""

from svc_app.screens.benchmark import BenchmarkScreen
from svc_app.screens.library import VoiceLibraryScreen
from svc_app.screens.projects import ProjectsScreen
from svc_app.screens.settings import SettingsScreen
from svc_app.screens.wizard import CoverWizard

__all__ = [
    "BenchmarkScreen",
    "CoverWizard",
    "ProjectsScreen",
    "SettingsScreen",
    "VoiceLibraryScreen",
]
