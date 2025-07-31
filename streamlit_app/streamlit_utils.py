# streamlit_app/streamlit_utils.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

import streamlit as st
import pandas as pd
import plotly.io as pio
from pathlib import Path
from streamlit_app.config import DATA_PATH, EXPORT_PLOTS, EXPORT_TABLES

# Ensure directories exist
EXPORT_PLOTS.mkdir(parents=True, exist_ok=True)
EXPORT_TABLES.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 📁 Dataset Loader
# -------------------------------
def load_dataset(filename: str) -> pd.DataFrame:
    """Load CSV file from synthetic data folder."""
    path = DATA_PATH / filename
    if not path.exists():
        st.error(f"❌ File not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"⚠️ Error loading {filename}: {e}")
        return pd.DataFrame()

# -------------------------------
# 💾 Save & Show Plot
# -------------------------------
def save_plot(fig, save_path: Path, format="png", show=True):
    """
    Save Plotly figure and display it in Streamlit.
    Format: 'png', 'svg', 'html'
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "html":
            pio.write_html(fig, file=save_path)
        elif format in ["png", "svg"]:
            pio.write_image(fig, file=save_path, format=format)
    except Exception as e:
        st.warning(f"⚠️ Could not save figure ({format}): {e}")

    if show:
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 📊 Display & Export Tables
# -------------------------------
def display_and_export_table(df: pd.DataFrame, save_path: Path, show_index=False):
    """
    Render table and export as CSV.
    Markdown export is optional to avoid 'tabulate' dependency errors.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Export CSV
        df.to_csv(save_path.with_suffix(".csv"), index=show_index)
        st.success(f"✅ Table exported to {save_path.with_suffix('.csv')}")
    except Exception as e:
        st.error(f"⚠️ Could not export CSV: {e}")

    st.dataframe(df)

# -------------------------------
# 📌 Sidebar Section UI
# -------------------------------
def sidebar_section(title: str, description: str = ""):
    """Create a stylized sidebar header with optional description."""
    st.sidebar.markdown(f"### {title}")
    if description:
        st.sidebar.caption(description)
