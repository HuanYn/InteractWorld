"""Operator-configured Creator guard; never creates GPU authorization."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.creator.guard import main


if __name__ == '__main__':
    raise SystemExit(main())
