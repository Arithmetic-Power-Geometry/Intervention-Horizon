# Intervention Horizon

Reference software for **Before Prediction Becomes Too Late: A Theory of Intervention Horizons for Preventable Harm**.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar. Licensed under Apache-2.0.

## GitHub workflow
Upload this folder as the root of a GitHub repository. Open **Actions -> One-click reproduce -> Run workflow**. The workflow installs the package, runs the full test suite, imports the interactive app as a smoke test, regenerates every benchmark table and figure, and uploads the results artifact.

## Local reproduction
```bash
python -m pip install -e '.[dev]'
pytest -q
python -m intervention_horizon.benchmark --out results --n 50000
```

## Interactive Research Explorer
```bash
python app.py
```
The Gradio app exposes expected hazard lead, hazard uncertainty, risk tolerance, fragility perturbation, ACT/CRITICAL thresholds, two intervention travel-time distributions, reliability and cost, time-costed value-of-information components, heterogeneous individual horizons, and shared-resource delay. It returns:

- viable interventions and success probabilities;
- action-specific last-safe departure times;
- Intervention Horizon, opportunity volume, fragility and decision status;
- an epsilon/safety-confidence sensitivity table and plot;
- exact TCVoI decomposition and OBSERVE/ACT-NOW decision;
- collective horizon and theorem-bound diagnostics.

The explorer is a research implementation, **not an operational emergency-warning service**.

## Reproducibility scope
The controlled experiments are fixed-seed theorem and stress tests. `results/tables/jharkhand_external_evidence.csv` contains literature-derived aggregate anchors with source labels; it is not a redistributed raw lightning-event dataset. Operational deployment requires event-level validated lightning, shelter, mobility, warning-receipt and behavioral data.
