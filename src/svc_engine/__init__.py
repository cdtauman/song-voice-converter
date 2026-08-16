"""SongVoice engine.

Hard rule enforced by tests: this package must never import PySide6 or any GUI
toolkit. The UI talks to it over JSON-RPC only.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
