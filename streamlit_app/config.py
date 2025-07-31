import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from pathlib import Path

# -----------------------------
# 📂 Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_PATH = PROJECT_ROOT / "synthetic_data" / "exports"
EXPORT_PLOTS = PROJECT_ROOT / "exports" / "plots"
EXPORT_TABLES = PROJECT_ROOT / "exports" / "tables"

# Streamlit pages folder (for dynamic page linking)
PAGES_PATH = PROJECT_ROOT / "streamlit_app" / "pages"

# -----------------------------
# 🎨 Theme & UI Settings
# -----------------------------
APP_TITLE = "🔬 SciPyMasterPro"
APP_ICON = "📊"
LAYOUT = "wide"

# Color palette (reusable across pages)
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#4f4f4f",
    "background": "#f9f9f9",
    "accent": "#ff7f0e",
    "success": "#2ca02c",
    "error": "#d62728"
}

# Default font sizes (can be used in custom HTML)
FONTS = {
    "title_size": "2rem",
    "subtitle_size": "1.3rem",
    "text_size": "1rem"
}

# -----------------------------
# 📦 File Handling Defaults
# -----------------------------
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_TABLE_FORMAT = "csv"
