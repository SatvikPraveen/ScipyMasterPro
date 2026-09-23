# syntax=docker/dockerfile:1.7
# =============================================================================
#   SciPyMasterPro image: JupyterLab + Streamlit
#
#   Build:   docker build -t scipymasterpro .
#   Run app: docker run -p 8501:8501 scipymasterpro
#   Run lab: docker run -p 8888:8888 scipymasterpro jupyter
#   Both:    docker run -p 8501:8501 -p 8888:8888 scipymasterpro both
# =============================================================================
ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------------------
# Stage 1: build wheels for every pinned dependency
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gfortran libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements_dev.txt .
RUN pip wheel --wheel-dir /wheels -r requirements_dev.txt

# ---------------------------------------------------------------------------
# Stage 2: slim runtime image
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="SciPyMasterPro" \
      org.opencontainers.image.description="Notebooks, utilities and an interactive app for mastering SciPy" \
      org.opencontainers.image.source="https://github.com/SatvikPraveen/ScipyMasterPro" \
      org.opencontainers.image.licenses="GPL-3.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    JUPYTER_PORT=8888 \
    STREAMLIT_PORT=8501

RUN apt-get update \
    && apt-get install -y --no-install-recommends libopenblas0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app

# Install the pre-built wheels (no compilers in the runtime image)
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-index --find-links=/wheels -r /wheels/requirements_dev.txt

# Copy the project and install it as a package so the CLI entry points exist
COPY --chown=app:app . .
RUN pip install --no-deps -e . \
    && mkdir -p exports/plots exports/tables synthetic_data/exports \
    && chown -R app:app /app

# JupyterLab configuration (token-less, for local/container use only)
USER app
RUN mkdir -p /home/app/.jupyter \
    && printf '%s\n' \
        "c.ServerApp.token = ''" \
        "c.ServerApp.password = ''" \
        "c.ServerApp.open_browser = False" \
        "c.ServerApp.allow_remote_access = True" \
        "c.ServerApp.ip = '0.0.0.0'" \
        > /home/app/.jupyter/jupyter_lab_config.py

EXPOSE 8501 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${STREAMLIT_PORT}/_stcore/health || exit 1

COPY --chown=app:app docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
CMD ["app"]
