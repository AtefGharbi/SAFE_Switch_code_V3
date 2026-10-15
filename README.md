# SAFE-Switch — code and results (V3)

Code for "SAFE-Switch: Certified Information-State Safety Filtering for Cyber-Physical Systems Under Hybrid Sensor
Attacks and Communication Faults" (V3, after the independent audit of Theorems 1–3).

Requirements: Python 3.11+ (tested 3.13), numpy, scipy, scikit-learn, matplotlib, pandas, cvxpy (Clarabel).

## Certificates (machine-readable)

    python verify_certificate.py dcmotor    # -> results/cert_verification_dcmotor.json
    python verify_certificate.py tank       # -> results/cert_verification_tank.json
    python audit_certificate.py             # independent recomputation -> results/cert_audit_<plant>.json
    python post_alarm.py                    # Proposition 2 quantities -> results/post_alarm.json

`cert_verification_<plant>.json` lists every quantity (rho, alpha, beta, c0, kappa, sigma, p, rbar, qbar, ell, lambda,
dA, dB, dE), each condition (C1)-(C5) with LHS, RHS, margin and pass/fail, the sampled one-step slack, the enclosure
check against the true plant at all parameter-box vertices, the corrected a-priori bound of Theorem 2, and the grace
window of Proposition 2(b). `cert_audit_<plant>.json` compares every number with an independently written
re-implementation (all agree to round-off) and checks the uncertainty bounds on a finer grid.

## Full reproduction

    ./v3_pipeline.sh          # everything below, in order (about 3 h on 2 cores)

Steps: risk models, threshold tuning, behaviour-cloning data, 25 PPO policies (MLP 8, GRU 6, CVaR 3, no-risk 2,
no-filter 2, tank 3), reproduction of Huang–Dong (2020), main matrix, ablation, worst-case supervisor, E12 supervisors,
E13 trade-off, E15 hardening, coupled tanks, compositional chain, scalability, learning curve, adversarial search
(6 runs), analysis (`analysis.py`, `analysis_v2.py` -> results/tables*.json) and figures. Additional check:

    python e5_infeas_check.py # every infeasible filter step in E5 coincides with a monitor alarm

`results/` contains every trained model and raw per-episode result (`*.jsonl`, `*.json`) used in the V3 manuscript, so
the analysis and figure steps can be run on their own. `results/v2_backup/` and `results/v1_backup/` hold the results
of the earlier versions (before the audit fixes; see Appendix B.1 of the manuscript).

Seeds: test 20000-20019 (paired across methods); risk model 1000-1159 / 5000-5059; BC 777000-777047; threshold tuning
600000-600023; validation 900000-900007.
