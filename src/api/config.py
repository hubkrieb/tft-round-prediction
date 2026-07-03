"""Static paths and board geometry shared across the API package."""

from pathlib import Path

# Board geometry: one side of the arena is 4 rows x 7 columns. Locations are
# encoded as "<row-letter><col-number>", e.g. A1..D7 (matches the raw data).
BOARD_ROWS = 4
BOARD_COLS = 7
MAX_ITEMS = 3
STAR_LEVELS = (1, 2, 3)

# Repository root (…/tft-round-prediction). All default paths are resolved
# relative to it so the API works regardless of the process CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Default trained models, kept together under models/. The ViT checkpoint is the
# one reported in the README (test accuracy 80.4%). Override any via the API / CLI.
DEFAULT_VIT_CKPT = "models/vit/vit.ckpt"
DEFAULT_CNN_CKPT = "models/cnn/cnn.ckpt"
DEFAULT_XGB_MODEL = "models/baseline/xgboost.json"

# Frontend lives in the sibling ``src/web`` package. The API serves it as static
# files, and ``fetch_assets`` writes the icons + catalog into it.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
ASSET_DIR = WEB_DIR / "assets"
CATALOG_PATH = WEB_DIR / "catalog.json"


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` against the repo root if it is relative."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
