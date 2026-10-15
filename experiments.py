"""Experiment runner: E1-E7 (DC motor), ablations (E6) and runtime/scalability (E8).

Usage:  python experiments.py main|ablation [n_test_seeds] [n_procs]
Results are written as JSON lines to results/<suite>.jsonl (one row per episode).
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scenarios import ScenarioCfg, sample
from sim import simulate, metrics

OUT = os.path.join(os.path.dirname(__file__), "results")
TEST_SEED0 = 20_000
_D = None
_RISK = None
_POL = {}


def scenarios_main():
    S = {}
    S["E1 nominal"] = dict()
    S["E2 mult (HD)"] = dict(attack="mult")
    for a in [0.1, 0.3, 1.0, 2.0]:
        S[f"E2 bias {a}"] = dict(attack="bias", attack_mag=a, single=True)
    S["E2 ramp 0.2/s"] = dict(attack="ramp", attack_mag=0.2, single=True)
    S["E2 stealth 0.03"] = dict(attack="stealth", attack_mag=0.03)
    for p in [0.1, 0.3, 0.5]:
        S[f"E3 loss {p}"] = dict(p_loss=p)
    S["E3 burst (uncapped)"] = dict(p_loss=0.1, burst=True, p_gb=0.05, p_bg=0.2, Nb_cap=None)
    S["E3 actuator loss 0.3"] = dict(pa_loss=0.3)
    for d in [1, 3, 6, 10]:
        S[f"E4 delay {d}"] = dict(dmax=d)
    net = dict(p_loss=0.2, burst=True, dmax=3, pa_loss=0.1)
    S["E5 bias+net"] = dict(attack="bias", attack_mag=1.0, single=True, **net)
    S["E5 mult+net"] = dict(attack="mult", **net)
    for u in [0.0, 0.03, 0.1, 0.2, 0.3]:
        S[f"E7 unc {u}"] = dict(attack="bias", attack_mag=1.0, single=True, rel_unc=u, **net)
    return S


def scenarios_stress():
    burst = dict(attack="bias", attack_mag=6.0, on_mean=0.1, min_on=0.1, off_mean=1.0)
    return {"E10 burst-bias 6 speed": dict(attack_channels=(1,), **burst),
            "E10 burst-bias 6 current": dict(attack_channels=(0,), **burst),
            "E10 burst-bias 6 speed+net": dict(attack_channels=(1,), p_loss=0.1, dmax=2, pa_loss=0.1, **burst),
            "E10 long bias 6 speed†": dict(attack="bias", attack_mag=6.0, attack_channels=(1,))}


ABL_SCEN = ["E1 nominal", "E2 bias 1.0", "E2 mult (HD)", "E3 loss 0.5", "E5 bias+net"]


def make_cfg(name, spec, seed):
    spec = dict(spec)
    single = spec.pop("single", False)
    if single:
        spec["attack_channels"] = (seed % 2,)
    return ScenarioCfg(name=name, **spec)


def _init():
    global _D, _RISK
    from config import Design
    from train import load_risk
    _D = Design()
    _RISK = load_risk()


def _policy(tag, s):
    from train import load_policy
    key = (tag, s)
    if key not in _POL:
        _POL[key] = load_policy(tag, s)[0]
    return _POL[key]


def build(method):
    from controllers import SafeSwitch, LQRKF, HinfSet, HuangDong2020, RobustMPC
    D, R = _D, _RISK
    name, _, arg = method.partition("@")
    if name == "LQR":
        return LQRKF(D)
    if name == "Robust H∞":
        return HinfSet(D)
    if name == "Robust MPC":
        return RobustMPC(D)
    if name == "Huang–Dong [1]":
        return HuangDong2020(D)
    if name == "Risk-threshold + filter":
        return SafeSwitch(D, "threshold", risk=R)
    if name == "Filter only (N)":
        return SafeSwitch(D, "always_N", risk=R)
    if name == "Robust mode + filter (R)":
        return SafeSwitch(D, "always_R", risk=R)
    if name == "Random supervisor + filter":
        c = SafeSwitch(D, "random", risk=R)
        return c
    if name == "Random supervisor w/o filter":
        return SafeSwitch(D, "random", risk=R, use_filter=False)
    if name == "Always N w/o filter":
        return SafeSwitch(D, "always_N", risk=R, use_filter=False)
    s = int(arg)
    if name == "SAFE-Switch":
        return SafeSwitch(D, "ppo", policy=_policy("full", s), risk=R)
    if name == "PPO w/o filter":
        return SafeSwitch(D, "ppo", policy=_policy("nofilter", s), risk=R, use_filter=False)
    if name == "w/o risk features":
        return SafeSwitch(D, "ppo", policy=_policy("norisk", s), risk=R, risk_in_obs=False)
    if name == "w/o timestamps":
        return SafeSwitch(D, "ppo", policy=_policy("full", s), risk=R, timestamp_aware=False)
    if name == "w/o dwell time":
        return SafeSwitch(D, "ppo", policy=_policy("full", s), risk=R, dwell=0)
    if name == "w/o packetized fallback":
        return SafeSwitch(D, "ppo", policy=_policy("full", s), risk=R, packetized=False)
    if name == "w/o zonotope (box only)":
        return SafeSwitch(D, "ppo", policy=_policy("full", s), risk=R, use_zono=False)
    if name == "w/o quarantine/re-anchoring":
        return SafeSwitch(D, "ppo", policy=_policy("full", s), risk=R, quarantine=False, reanchor=False)
    raise ValueError(method)


def task(args):
    scen, spec, seed, method = args
    spec = dict(spec)
    cfg = make_cfg(scen, spec, seed)
    sc = sample(cfg, TEST_SEED0 + seed, xmax=_D.xmax)
    ctrl = build(method)
    if hasattr(ctrl, "rng"):
        ctrl.rng = np.random.default_rng(seed)
    t0 = time.time()
    out = simulate(_D, ctrl, sc)
    m = metrics(_D, out)
    name, _, ps = method.partition("@")
    m.update(scenario=scen, seed=seed, method=name, policy_seed=int(ps) if ps else -1,
             wall=time.time() - t0)
    return m


def run(suite, methods, scen_names, n_seeds, procs):
    S = {**scenarios_main(), **scenarios_stress()}
    tasks = [(s, S[s], i, m) for s in scen_names for i in range(n_seeds) for m in methods]
    path = os.path.join(OUT, f"{suite}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            ps = r["policy_seed"]
            done.add((r["scenario"], r["seed"], r["method"] + (f"@{ps}" if ps >= 0 else "")))
    tasks = [t for t in tasks if (t[0], t[2], t[3]) not in done]
    # expensive tasks first for better load balance
    tasks.sort(key=lambda t: 0 if "MPC" in t[3] else 1)
    print(f"{suite}: {len(tasks)} tasks", flush=True)
    t0 = time.time()
    with Pool(procs, initializer=_init) as pool, open(path, "a") as f:
        for i, r in enumerate(pool.imap_unordered(task, tasks, chunksize=2)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    suite = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    if suite == "main":
        methods = ["LQR", "Robust H∞", "Robust MPC", "Huang–Dong [1]", "Risk-threshold + filter"] + \
                  [f"SAFE-Switch@{s}" for s in range(5)]
        if len(sys.argv) > 4 and sys.argv[4] == "withnf":
            methods += [f"PPO w/o filter@{s}" for s in range(2)]
        run("main", methods, list(scenarios_main().keys()), n, procs)
    elif suite == "stress":
        methods = ["LQR", "Robust H∞", "Robust MPC", "Huang–Dong [1]", "Filter only (N)", "Always N w/o filter",
                   "Random supervisor + filter", "Random supervisor w/o filter", "Risk-threshold + filter"] + \
                  [f"SAFE-Switch@{s}" for s in range(5)] + [f"PPO w/o filter@{s}" for s in range(2)]
        run("stress", methods, list(scenarios_stress().keys()), n, procs)
    elif suite == "ablation":
        methods = ["Filter only (N)", "Robust mode + filter (R)", "Random supervisor + filter"] + \
                  [f"{v}@{s}" for v in ["w/o risk features"] for s in range(2)] + \
                  [f"{v}@{s}" for v in ["w/o timestamps", "w/o dwell time", "w/o packetized fallback",
                                        "w/o zonotope (box only)", "w/o quarantine/re-anchoring"]
                   for s in range(3)]
        run("ablation", methods, ABL_SCEN, n, procs)
