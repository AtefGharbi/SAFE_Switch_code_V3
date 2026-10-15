"""V2 experiment suites.

  e12   : supervisor study (MLP-PPO x10, GRU-PPO x10, CVaR-PPO x5, default/tuned threshold, fixed R,
          always N) - identical filtered action pipeline and test trajectories
  e15   : threat-model hardening (persistent-attack recovery, authenticated beacons, white-box
          gate-riding attack) incl. baselines on the new attack
  tank  : second certified benchmark (coupled tanks)
  e13   : safety-performance trade-off (contraction rate x estimator conservatism)

Usage: python experiments_v2.py <suite> [n_seeds] [procs]   -> results/<suite>.jsonl
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scenarios import ScenarioCfg, sample
from sim import simulate, metrics
from experiments import scenarios_main, scenarios_stress, TEST_SEED0
from bench_designs import make_design, PROFILES

OUT = os.path.join(os.path.dirname(__file__), "results")
_DES, _RISK, _POL, _TUNED = {}, {}, {}, None


def scen_dict(bench):
    S = {**scenarios_main(), **scenarios_stress()}
    S["E11 gate-riding (white-box)"] = dict(attack="gate", attack_mag=3.0, single=True)
    S["E11 gate-riding+net"] = dict(attack="gate", attack_mag=3.0, single=True, p_loss=0.2, burst=True, dmax=3,
                                    pa_loss=0.1)
    prof = PROFILES[bench]
    out = {}
    for k, v in S.items():
        v = dict(v)
        v.update(T=prof["T"], Ts=prof["Ts"], time_scale=prof["time_scale"], w_scale=prof["w_scale"])
        out[k] = v
    return out


def make_cfg(name, spec, seed):
    spec = dict(spec)
    if spec.pop("single", False):
        spec["attack_channels"] = (seed % 2,)
    spec.pop("auth", None)
    return ScenarioCfg(name=name, **spec)


def design(bench, **kw):
    key = (bench, tuple(sorted(kw.items())))
    if key not in _DES:
        _DES[key] = make_design(bench, **kw)
    return _DES[key]


def risk(bench):
    from train import load_risk
    if bench not in _RISK:
        p = os.path.join(OUT, "risk_model.json" if bench == "dcmotor" else f"risk_model_{bench}.json")
        _RISK[bench] = load_risk(p)
    return _RISK[bench]


def policy(kind, s):
    from train2 import load
    from train import load_policy
    key = (kind, s)
    if key not in _POL:
        _POL[key] = load_policy("tank", s)[0] if kind == "tank" else load(kind, s)
    return _POL[key]


def tuned():
    global _TUNED
    if _TUNED is None:
        _TUNED = json.load(open(os.path.join(OUT, "tuned_threshold.json")))["best"]
    return _TUNED


def build(method, bench, dkw):
    """method string: '<name>[@seed]'; returns (controller, auth_period or None)."""
    from controllers import SafeSwitch, LQRKF, HinfSet, HuangDong2020, RobustMPC
    D, R = design(bench, **dkw), risk(bench)
    name, _, arg = method.partition("@")
    s = int(arg) if arg else None
    ppo_kind = "tank" if bench == "tank" else "mlp"
    simple = {
        "LQR": lambda: LQRKF(D), "Robust H∞": lambda: HinfSet(D), "Robust MPC": lambda: RobustMPC(D),
        "Huang–Dong [1]": lambda: HuangDong2020(D),
        "Threshold + filter": lambda: SafeSwitch(D, "threshold", risk=R),
        "Fixed R + filter": lambda: SafeSwitch(D, "always_R", risk=R),
        "Always N + filter": lambda: SafeSwitch(D, "always_N", risk=R),
        "Always N w/o filter": lambda: SafeSwitch(D, "always_N", risk=R, use_filter=False),
    }
    if name in simple:
        return simple[name](), None
    if name == "Tuned threshold + filter":
        t = tuned()
        return SafeSwitch(D, "threshold", risk=R, thr_on=t["thr_on"], thr_off=t["thr_off"], unc_on=t["unc_on"]), None
    if name == "MLP-PPO":
        return SafeSwitch(D, "ppo", policy=policy(ppo_kind, s), risk=R), None
    if name == "GRU-PPO":
        return SafeSwitch(D, "ppo", policy=policy("gru", s), risk=R), None
    if name == "CVaR-PPO":
        return SafeSwitch(D, "ppo", policy=policy("cvar", s), risk=R), None
    # hardening variants (supervisor: default risk-threshold rule)
    if name == "Threshold + PR":
        return SafeSwitch(D, "threshold", risk=R, gross_emax=1.0, recovery=True), None
    if name.startswith("Threshold + Auth"):
        M = int(name.split("Auth")[1].split()[0])
        pr = "PR" in name
        return SafeSwitch(D, "threshold", risk=R, auth_period=M, gross_emax=1.0 if pr else None,
                          recovery=pr), M
    if name == "MLP-PPO + PR":
        return SafeSwitch(D, "ppo", policy=policy(ppo_kind, s), risk=R, gross_emax=1.0, recovery=True), None
    raise ValueError(method)


def task(args):
    suite, bench, scen, spec, seed, method, dkw = args
    D = design(bench, **dkw)
    cfg = make_cfg(scen, spec, seed)
    sc = sample(cfg, TEST_SEED0 + seed, xmax=D.xmax)
    ctrl, M = build(method, bench, dkw)
    if M:
        sc["auth_period"] = M
    if hasattr(ctrl, "rng"):
        ctrl.rng = np.random.default_rng(seed)
    t0 = time.time()
    out = simulate(D, ctrl, sc)
    m = metrics(D, out)
    name, _, ps = method.partition("@")
    # a-posteriori Theorem-2 check (13) on the realised trajectory
    if "C" in out and np.isfinite(out["F"]).any():
        C = D.cert; Cc, Rr, U = out["C"], out["R"], out["U"]
        s_ = np.array([C.normP(c) for c in Cc])
        b = s_[0]; ok = True; bmax_tail = 0.0
        # V3 (corrected Theorem 2): at steps where the actuator applies a buffered input K_R c_hat,
        # ||A c + B K_R c_hat||_P <= rho ||c||_P + eta with eta <= b_K^T rbar, b_K,i = ||P^1/2 B K_R e_i||
        eta_buf = float(np.linalg.norm(C.Ph @ C.B @ C.K, axis=0) @ C.rbar)
        BUF = out.get("BUF", np.zeros(len(U), bool))
        # Theorem 2 covers steps under Theorem 1 / Corollary 1 only: check the prefix before the first alarm
        AL = np.where(out.get("ALARM", np.zeros(len(U), bool)))[0]
        n_chk = int(AL[0]) if len(AL) else len(U) - 1
        for k in range(min(n_chk, len(U) - 1)):
            _, rp = C.predict(Cc[k], Rr[k], U[k:k + 1])
            g = float(C.p @ (rp - Rr[k + 1]))
            b = C.lam * b + C.eps0 + max(g, 0.0) + (eta_buf if BUF[k] else 0.0)
            ok &= bool(s_[k + 1] <= b + 1e-9)
            if k > len(U) // 2:
                bmax_tail = max(bmax_tail, b)
        m.update(thm2_holds=float(ok), thm2_bound_tail=float(bmax_tail),
                 thm2_frac_checked=float(min(n_chk, len(U) - 1) / (len(U) - 1)))
    m.update(suite=suite, bench=bench, scenario=scen, seed=seed, method=name,
             policy_seed=int(ps) if ps else -1, dkw=json.dumps(dkw), wall=time.time() - t0)
    return m


def run(suite, tasks, procs):
    path = os.path.join(OUT, f"{suite}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ps = r["policy_seed"]
            done.add((r["scenario"], r["seed"], r["method"] + (f"@{ps}" if ps >= 0 else ""), r["dkw"]))
    tasks = [t for t in tasks if (t[2], t[4], t[5], json.dumps(t[6])) not in done]
    tasks.sort(key=lambda t: 0 if "MPC" in t[5] else 1)
    print(f"{suite}: {len(tasks)} tasks", flush=True)
    t0 = time.time()
    with Pool(procs) as pool, open(path, "a") as f:
        for i, r in enumerate(pool.imap_unordered(task, tasks, chunksize=2)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)


E12_SCEN = ["E1 nominal", "E2 mult (HD)", "E2 bias 0.3", "E2 bias 2.0", "E2 ramp 0.2/s", "E3 loss 0.5",
            "E4 delay 3", "E5 bias+net", "E5 mult+net", "E10 burst-bias 6 speed"]
E15_SCEN = ["E1 nominal", "E2 mult (HD)", "E2 bias 1.0", "E2 ramp 0.2/s", "E5 bias+net", "E5 mult+net",
            "E10 long bias 6 speed†", "E11 gate-riding (white-box)", "E11 gate-riding+net"]
TANK_SCEN = ["E1 nominal", "E2 mult (HD)", "E2 bias 1.0", "E3 loss 0.5", "E4 delay 3", "E5 bias+net",
             "E10 burst-bias 6 speed", "E10 long bias 6 speed†", "E11 gate-riding (white-box)"]

if __name__ == "__main__":
    suite = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    T = []
    if suite == "e12":
        S = scen_dict("dcmotor")
        meths = ([f"MLP-PPO@{s}" for s in range(8)] + [f"GRU-PPO@{s}" for s in range(6)] +
                 [f"CVaR-PPO@{s}" for s in range(3)] +
                 ["Threshold + filter", "Tuned threshold + filter", "Fixed R + filter", "Always N + filter"])
        T = [(suite, "dcmotor", sc, S[sc], i, m, {}) for sc in E12_SCEN for i in range(n) for m in meths]
    elif suite == "e15":
        S = scen_dict("dcmotor")
        meths = ["Threshold + filter", "Threshold + PR", "Threshold + Auth2", "Threshold + Auth3",
                 "Threshold + Auth5", "Threshold + Auth3 PR"]
        T = [(suite, "dcmotor", sc, S[sc], i, m, {}) for sc in E15_SCEN for i in range(n) for m in meths]
        for sc in ["E11 gate-riding (white-box)", "E11 gate-riding+net"]:
            T += [(suite, "dcmotor", sc, S[sc], i, m, {}) for i in range(n)
                  for m in ["LQR", "Robust H∞", "Robust MPC", "Huang–Dong [1]", "Fixed R + filter"]]
    elif suite == "tank":
        S = scen_dict("tank")
        meths = ["LQR", "Robust H∞", "Robust MPC", "Huang–Dong [1]", "Threshold + filter", "Fixed R + filter",
                 "Always N + filter", "Always N w/o filter", "Threshold + PR", "Threshold + Auth3"] + \
                [f"MLP-PPO@{s}" for s in range(3)]
        T = [(suite, "tank", sc, S[sc], i, m, {}) for sc in TANK_SCEN for i in range(n) for m in meths]
    elif suite == "e13":
        S = scen_dict("dcmotor")
        from config import Design
        rho = float(np.ceil(Design().cert.rho * 1e4) / 1e4)   # smallest 4-decimal value >= rho (C5)
        for lam in [rho, 0.92, 0.94, 0.96, 0.98]:
            for abar in [0.03, 0.06]:
                dkw = dict(lam=lam, abar=abar)
                for sc in ["E1 nominal", "E3 loss 0.5", "E5 bias+net", "E2 mult (HD)"]:
                    for m in ["Tuned threshold + filter", "Always N + filter"]:
                        T += [(suite, "dcmotor", sc, S[sc], i, m, dkw) for i in range(n)]
    run(suite, T, procs)
