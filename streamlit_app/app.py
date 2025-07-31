import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

import streamlit as st

# -----------------------------
# 🎨 Page Config
# -----------------------------
st.set_page_config(
    page_title="🧠 SciPyMasterPro",
    layout="wide",
    page_icon="📊"
)

# -----------------------------
# 🌟 Custom CSS for Styling
# -----------------------------

st.markdown("""
    <style>
    /* Sidebar background and text color */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;  /* light gray background */
        color: #000000 !important;             /* dark text */
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] .css-1d391kg, 
    section[data-testid="stSidebar"] .css-1v3fvcr,
    section[data-testid="stSidebar"] div, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {
        color: #222222 !important;
        font-weight: 500 !important;
    }

    /* Sidebar titles */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #1f77b4 !important;
        font-weight: 700 !important;
    }

    /* Sidebar divider line */
    section[data-testid="stSidebar"] hr {
        border: none;
        height: 1px;
        background-color: #ccc;
        margin: 0.5rem 0;
    }

    /* Sidebar hover effect for links */
    a {
        color: #1f77b4 !important;
    }
    a:hover {
        color: #ff4b4b !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# 🏷️ Title and Description
# -----------------------------
st.markdown("<h1 class='main-title'>🔬 SciPyMasterPro</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='description'>
Welcome to <b>SciPyMasterPro</b> – an interactive, modular learning platform to master <br>
<b>statistics, simulation, and distribution fitting</b> using <b>SciPy</b> & <b>Statsmodels</b>. <br><br>
👉 <i>Navigate using the sidebar to explore statistical modules.</i>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 🎛 Sidebar Navigation
# -----------------------------
st.sidebar.title("🧭 Modules Navigation")
st.sidebar.markdown("---")
st.sidebar.page_link("pages/shared_pdf_ecdf.py", label="📈 Shared: PDF & ECDF")
st.sidebar.page_link("pages/03_distribution_fitting.py", label="📊 Distribution Fitting")
