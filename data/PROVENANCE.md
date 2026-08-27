# Data provenance

- Controlled benchmark, matched-risk, observe-or-act, collective, sensitivity, and ablation datasets are generated deterministically by `src/intervention_horizon/benchmark.py` using fixed seeds.
- Jharkhand district population totals are from Census of India 2011 district tables.
- Literature-derived empirical anchors are explicitly labeled in `results/tables/jharkhand_external_evidence.csv` and correspond to Mishra et al. (2025, *Natural Hazards*, DOI 10.1007/s11069-025-07124-3) and IMD's Annual Climate Summary 2025.
- No IITM/NRSC event-level lightning stream or individual casualty record is redistributed.
