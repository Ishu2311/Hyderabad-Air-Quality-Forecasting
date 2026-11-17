"""
Root Dashboard Launcher
Runs the Hyderabad PM2.5 Research Dashboard application.
"""

import sys
from pathlib import Path

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PROJECT_ROOT.parent

for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.app import main

if __name__ == "__main__":
    main()

