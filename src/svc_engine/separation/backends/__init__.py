"""Concrete separation backends, all satisfying `SeparationBackend`."""

from svc_engine.separation.backends.audio_separator_backend import AudioSeparatorBackend
from svc_engine.separation.backends.pymss_backend import PymssBackend

#: Preference order. audio-separator first: wider model coverage and far more
#: real-world use. pymss is the independent second source, which matters because
#: model hosting disappears (see docs/phase-reports/phase-2.md).
BACKEND_ORDER = ("audio_separator", "pymss")

__all__ = ["AudioSeparatorBackend", "PymssBackend", "BACKEND_ORDER"]
