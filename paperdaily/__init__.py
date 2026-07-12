"""Local-first arXiv discovery and deep-reading extensions for PaperFlow."""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

# PaperFlow's current source layout keeps operational modules in top-level
# ``skills/``, ``agents/`` and ``deployments/`` directories rather than in the
# installable ``paperflow`` package.  Editable console-script launches start
# with ``.venv/Scripts`` on sys.path, so add the checkout root when those
# directories are present.  This is deliberately scoped to source installs.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if (PROJECT_ROOT / "skills").is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["PROJECT_ROOT", "__version__"]
