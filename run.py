"""Run one stage of the study from the repository folder without installing the package.

Examples:
    python run.py --stage audit --local
    python run.py --stage main --local --work-dir /data/work
    python run.py --stage analysis --local
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from whose_emotion.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
