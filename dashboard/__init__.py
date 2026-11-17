"""
Hyderabad PM2.5 Research Dashboard Package
"""

import sys
from pathlib import Path

# Ensure new_research_project (project root) and repo root are in sys.path
_DASHBOARD_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DASHBOARD_DIR.parent
_REPO_ROOT = _PROJECT_ROOT.parent

for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
