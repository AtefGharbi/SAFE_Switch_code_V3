"""Tuned risk-threshold supervisor: grid search of (thr_on, thr_off, unc_on) on the PPO training
distribution with the PPO reward (same objective, same filtered actions)."""
import itertools, json, os
import numpy as np
from multiprocessing import Pool
from config import Design
from controllers import SafeSwitch
from sim import simulate
from train import load_risk, make_train_sc, reward, OUT

SEEDS = list(range(600_000, 600_024))
_D = _R = None


def _init():
    global _D, _R
    _D = Design(); _R = load_risk()


def evalcfg(p):
    on, off, unc = p
    J = []
    for s in SEEDS:
        cfg, sc = make_train_sc(_D, s)
        c = SafeSwitch(_D, "threshold", risk=_R, thr_on=on, thr_off=off, unc_on=unc)
        J.append(-simulate(_D, c, sc, reward_fn=reward)["RW"].sum())
    return dict(thr_on=on, thr_off=off, unc_on=unc, cost=float(np.mean(J)), cost_median=float(np.median(J)))


if __name__ == "__main__":
    grid = [(on, round(on - d, 2), unc) for on in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for d in [0.1, 0.2, 0.3]
            for unc in [0.4, 0.6, 0.8, 1.01] if on - d > 0]
    with Pool(2, initializer=_init) as P:
        res = P.map(evalcfg, grid)
    res.sort(key=lambda r: r["cost"])
    json.dump(dict(best=res[0], default=[r for r in res if (r["thr_on"], r["thr_off"], r["unc_on"]) == (0.5, 0.3, 0.6)],
                   all=res), open(os.path.join(OUT, "tuned_threshold.json"), "w"), indent=1)
    print(res[:5])
