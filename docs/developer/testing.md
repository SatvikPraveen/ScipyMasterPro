# Testing

| Command | What runs |
| --- | --- |
| `make test` | unit tests, Hypothesis property tests, plot smoke tests (`-m "not app"`) |
| `make test-app` | every Streamlit page through `AppTest` (`-m app`) |
| `make test-notebooks` | every notebook through `nbmake` |
| `make test-all` | all of the above |
| `make test-cov` | unit tests with an HTML coverage report |

## Layers

- **Unit tests** (`tests/test_*_utils.py`): behaviour of each utility module.
- **Property tests** (`tests/test_properties.py`): invariants such as "an ECDF is monotone
  and bounded" or "Cohen's d is antisymmetric", searched by Hypothesis.
- **Plot smoke tests** (`tests/test_viz_utils.py`): every plotting helper renders on the Agg
  backend and returns a figure.
- **App tests** (`tests/test_streamlit_app.py`): every page renders without raising.
- **CLI tests** (`tests/test_cli.py`).
- **Notebooks**: executed in CI so examples never rot.

## Markers

`unit`, `integration`, `slow`, `app`, `property`. Select with `pytest -m <marker>`.

## Writing a test

Fixtures for common datasets, matrices and tolerances live in `tests/conftest.py`. Prefer a
property test when the behaviour is an invariant, a unit test when it is a specific value.
