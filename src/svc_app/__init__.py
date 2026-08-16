"""SongVoice UI.

Hard rule enforced by tests: this package must never import torch or any AI
library. It talks to the engine process over JSON-RPC.

The PySide6 window arrives in Phase 8. Phase 1 ships the engine client only.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
