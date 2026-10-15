"""E9: sensitivity to the contraction rate lambda of the safety filter (Theorem 3 trade-off)."""
import json, os, sys
import numpy as np
from multiprocessing import Pool
from config import Design
from scenarios import sample
from sim import simulate, metrics
from experiments import scenarios_main, make_cfg, TEST_SEED0

OUT = os.path.join(os.path.dirname(__file__), "results")
SC = ["E1 nominal", "E3 loss 0.5", "E5 bias+net"]


_CACHE = {}


def task(args):
    lam, scen, seed, sup = args
    from controllers import SafeSwitch
    from train import load_risk
    if lam not in _CACHE:
        _CACHE[lam] = Design(lam=lam)
    D = _CACHE[lam]
    cfg = make_cfg(scen, scenarios_main()[scen], seed)
    sc = sample(cfg, TEST_SEED0 + seed, xmax=D.xmax)
    out = simulate(D, SafeSwitch(D, sup, risk=load_risk()), sc)
    m = metrics(D, out)
    # empirical information gain gamma_k = p'(r+_k - r_{k+1}) and the Theorem-3 bound
    C = D.cert
    R = out["R"]; Cc = out["C"]; U = out["U"]
    g = []
    for k in range(len(U) - 1):
        _, rp = C.predict(Cc[k], R[k], U[k:k + 1])
        g.append(float(C.p @ (rp - R[k + 1])))
    g = np.array(g)
    m.update(lam=lam, scenario=scen, seed=seed, sup=sup, gamma_mean=float(np.mean(g)),
             gamma_p95=float(np.percentile(g, 95)),
             bound=float((C.eps0 + np.percentile(g, 95)) / (1 - lam)),
             sP_tail=float(np.mean([C.normP(c) for c in Cc[len(Cc) // 2:]])))
    return m


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    rho = Design().cert.rho
    lams = [round(rho, 4), 0.94, 0.98]
    tasks = [(l, s, i, sup) for l in lams for s in SC for i in range(n) for sup in ["always_N", "threshold"]]
    with Pool(2) as p:
        res = p.map(task, tasks, chunksize=4)
    json.dump(res, open(os.path.join(OUT, "lam_sweep.json"), "w"))
    print("done", len(res))
