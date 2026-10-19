"""One immutable fixed-N round; preview by default, --run needs --budget-sec."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.fixed_n.round import main
if __name__ == "__main__":
    main()
