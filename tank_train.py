"""Benchmark 2 (coupled tanks): risk model + PPO supervisors (same recipe as the DC motor)."""
import json, os, sys
from bench_designs import make_design
from train import train_risk, train_ppo, load_risk, OUT

if __name__ == "__main__":
    D = make_design("tank")
    p = os.path.join(OUT, "risk_model_tank.json")
    if not os.path.exists(p):
        res = train_risk(D, n_train=160, n_test=60)
        json.dump(res, open(p, "w"), indent=1)
        print({k: res[k] for k in ("auc", "brier", "ece", "tpr", "fpr")}, flush=True)
    risk = load_risk(p)
    for s in [int(x) for x in sys.argv[1].split(",")]:
        train_ppo(D, risk, s, iters=30, tag="tank")
