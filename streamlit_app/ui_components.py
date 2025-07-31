import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

import streamlit as st
import pandas as pd
from streamlit_app.config import DATA_PATH

# ==================================================
# 📂 Load Synthetic Datasets
# ==================================================
@st.cache_data(show_spinner=False)
def load_synthetic_distribution_data():
    """Load synthetic distribution data from CSV."""
    file_path = DATA_PATH / "mixed_distributions.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_time_series_data():
    """Load synthetic time series dataset (placeholder)."""
    file_path = DATA_PATH / "timeseries.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.warning("⚠️ Time series dataset not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")
        return pd.DataFrame()


# ==================================================
# 🎛️ Sidebar Components
# ==================================================
def select_distribution_column(df: pd.DataFrame):
    """Sidebar dropdown to choose a distribution column."""
    if df.empty:
        st.sidebar.error("No data available to select a column.")
        return None
    columns = df.columns.tolist()
    return st.sidebar.selectbox("📂 **Choose a Distribution Column**", columns)

def add_sidebar_notes():
    """Sidebar info block for guidance."""
    with st.sidebar:
        st.markdown("### ℹ️ Notes")
        st.markdown("📘 *Choose a distribution column to visualize its empirical and theoretical fit.*")
        st.markdown("🧪 Includes ECDF, PDF overlay, and test statistics.")


# ==================================================
# 🌟 Shared Page Setup (from base_page.py)
# ==================================================
def setup_page(title: str, icon="🧠", layout="wide"):
    """Set page configuration and apply a consistent style."""
    st.set_page_config(page_title=title, page_icon=icon, layout=layout)
    st.markdown(
        f"""
        <div style="text-align:center; padding:10px; background-color:#1f2937; border-radius:8px;">
            <h1 style="color:#f3f4f6;">{icon} {title}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

def setup_sidebar(module_title: str, description: str = ""):
    """Create a consistent sidebar structure."""
    st.sidebar.markdown("### 🧭 Navigation")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {module_title}")
    if description:
        st.sidebar.info(description)

def handle_missing_data(df, msg="⚠️ No data available for this module."):
    """Gracefully handle empty datasets."""
    if df is None or df.empty:
        st.warning(msg)
        st.stop()

def add_footer():
    """Add a footer for branding and navigation consistency."""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; font-size:14px; color:gray;">
            🚀 <b>SciPyMasterPro</b> | Interactive Statistical Learning |
            <a href="https://github.com/your-repo" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )
