"""Check that every infeasible filter step in E5 (SAFE-Switch, 5 policies x 20 seeds) occurs while the runtime
monitor reports an assumption violation (M1-M3), i.e. outside the certified envelope."""
import os, numpy as np
import experiments as X
from scenarios import sample
from sim import simulate
X._init(); D = X._D
spec = X.scenarios_main()["E5 bias+net"]
tot = cov = m1 = 0
for seed in range(20):
    for ps in range(5):
        cfg = X.make_cfg("E5 bias+net", spec, seed); sc = sample(cfg, X.TEST_SEED0 + seed, xmax=D.xmax)
        c = X.build(f"SAFE-Switch@{ps}"); orig = c.step; log = []
        def st(k, arr, u, orig=orig, c=c):
            out = orig(k, arr, u); i = out[2]; cc, rr = i["c"], i["r"]
            gap = k - np.min(c.est.last_tau)
            log.append((i["feas"], gap > D.Nb + D.dmax, bool(np.any(rr > c.D.cert.rbar + 1e-9)) or c.D.cert.f(cc, rr) > D.ell + 1e-9))
            return out
        c.step = st; simulate(D, c, sc)
        for f, a1, a23 in log:
            if not f:
                tot += 1; cov += (a1 or a23); m1 += a1
why = "with the age alarm (M1) active in every case" if m1 == tot else f"with (M1) active in {m1} cases and (M2)–(M3) in the others"
txt = (f"we checked all {tot} of them: every one occurred while the monitor reported an assumption violation, "
       f"{why}, that is, outside the certified envelope." if tot and cov == tot else
       (f"we checked all {tot} of them: {cov} occurred while the monitor reported (M1)–(M3)." if tot else
        "no infeasible step occurred."))
open(os.path.join("results", "e5_infeas_check.txt"), "w").write(txt)
print(tot, cov, m1, txt)
