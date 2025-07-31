# ===============================
#   SciPyMasterPro Dockerfile
# ===============================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (includes jupyterlab)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyterlab

# Copy project files
COPY . .

# ------------------------------
# Configure Jupyter
# ------------------------------
# Disable token and browser auto-open
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_lab_config.py

# Expose ports
# Streamlit
EXPOSE 8501
# JupyterLab   
EXPOSE 8888   

# ------------------------------
# Run both Streamlit and Jupyter
# ------------------------------
CMD bash -c "\
    streamlit run streamlit_app/app.py --server.port=8501 --server.headless=true & \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
"
