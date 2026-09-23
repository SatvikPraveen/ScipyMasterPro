#!/usr/bin/env bash
# Container entrypoint: choose which service(s) to run.
#   app      -> Streamlit only (default)
#   jupyter  -> JupyterLab only
#   both     -> Streamlit + JupyterLab in one container
#   anything else is executed verbatim (e.g. `pytest`, `bash`)
set -euo pipefail

STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

run_streamlit() {
    exec streamlit run streamlit_app/app.py \
        --server.port="${STREAMLIT_PORT}" \
        --server.address=0.0.0.0 \
        --server.headless=true
}

run_jupyter() {
    exec jupyter lab --ip=0.0.0.0 --port="${JUPYTER_PORT}" --no-browser
}

case "${1:-app}" in
    app)
        run_streamlit
        ;;
    jupyter)
        run_jupyter
        ;;
    both)
        jupyter lab --ip=0.0.0.0 --port="${JUPYTER_PORT}" --no-browser &
        run_streamlit
        ;;
    *)
        exec "$@"
        ;;
esac
