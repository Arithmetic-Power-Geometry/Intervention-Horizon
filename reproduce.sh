#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e '.[dev]'
pytest -q
python -c "import app; assert app.demo is not None"
python -m intervention_horizon.benchmark --out results --n 50000
