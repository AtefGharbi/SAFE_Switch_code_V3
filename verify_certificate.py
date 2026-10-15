"""Machine-readable verification of the Theorem-1/2 certificate for a Design.

Writes results/cert_verification_<bench>.json with every quantity, each condition
(LHS, RHS, margin, pass) and a Monte-Carlo/vertex check of the key inequality
f(Pi(c, r, K_R c)) <= ell on 200 000 sampled information states (c, r) in I.
"""
import json, os, sys, itertools
import numpy as np
from config import Design

OUT = os.path.join(os.path.dirname(__file__), "results")


def verify(D, n_samples=200_000, seed=0):
    C = D.cert
    ell = D.ell
    a, b, c0, q = C.alpha, C.beta, C.c0, C.qbar
    conds = {
        "C1_alpha_lt_1": dict(lhs=a, rhs=1.0),
        "C2_level_invariance": dict(lhs=a * ell + max(b - a, 0) * q + c0, rhs=ell),
        "C3_qbar_lt_ell": dict(lhs=q, rhs=ell),
        "C4a_input": dict(lhs=C.kap * ell, rhs=D.ufilt),        # kappa*ell <= ubar (filter input bound)
        "C4b_state": dict(lhs=float(np.max(C.sig * ell / D.xmax)), rhs=1.0),
        "C5_lambda_ge_rho": dict(lhs=C.rho, rhs=C.lam),
    }
    for k, v in conds.items():
        v["margin"] = v["rhs"] - v["lhs"]
        v["pass"] = bool(v["margin"] >= 0 if k != "C1_alpha_lt_1" and k != "C3_qbar_lt_ell" else v["margin"] > 0)
    # sampled check of the one-step inequality behind Theorem 1
    rng = np.random.default_rng(seed)
    n = D.plant.n
    worst = -np.inf
    worst_contr = -np.inf
    Ph = C.Ph
    Phi = np.linalg.inv(Ph)
    for _ in range(n_samples // 1000):
        R = rng.uniform(0, 1, (1000, n)) ** 0.3 * C.rbar  # biased towards rbar
        q_s = R @ C.p
        s_max = np.maximum(ell - q_s, 0)
        dirs = rng.normal(size=(1000, n)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        s = s_max * rng.uniform(0.8, 1.0, 1000) ** 0.2
        Cc = (Phi @ (dirs * s[:, None]).T).T
        for c, r in zip(Cc, R):
            if C.f(c, r) > ell + 1e-12:
                continue
            u = C.K @ c
            _, rp = C.predict(c, r, u)
            fval = C.normP(C.A @ c + C.B @ u) + C.p @ rp
            worst = max(worst, fval - ell)
            worst_contr = max(worst_contr, C.normP(C.A @ c + C.B @ u) - C.lam * C.normP(c))
    # V3: enclosure check against the TRUE plant: physical parameters at all vertices of the parameter box
    # plus random interior points (so dA, dB and dE are all exercised), box vertices for x, extreme w
    pl = D.plant
    params = []
    for scal in itertools.product([1 - D.rel_unc, 1 + D.rel_unc], repeat=len(pl.unc_keys)):
        params.append(dict(pl.nominal, **{k_: pl.nominal[k_] * s_ for k_, s_ in zip(pl.unc_keys, scal)}))
    for _ in range(64):
        params.append(dict(pl.nominal, **{k_: pl.nominal[k_] * (1 + D.rel_unc * rng.uniform(-1, 1))
                                          for k_ in pl.unc_keys}))
    true_mats = [pl.matrices(pp) for pp in params]
    enclosure_ok, worst_encl = True, -np.inf
    xv = np.array(list(itertools.product([-1, 1], repeat=n)), float)
    for _ in range(300):
        s0 = rng.uniform(0, 1) * ell
        d0 = rng.normal(size=n); d0 /= np.linalg.norm(d0)
        c = Phi @ (d0 * s0 * 0.9)
        r = rng.uniform(0, 1, n) * C.rbar
        u = np.atleast_1d(np.clip(rng.uniform(-1, 1, D.plant.m) * D.ufilt, -D.ufilt, D.ufilt))
        cp, rp = C.predict(c, r, u)
        for (At, Bt, Et) in true_mats:
            for sv in xv:
                x = c + r * sv
                for ws in (-1.0, 1.0):
                    xn = At @ x + Bt @ u + Et @ (C.wbar * ws)
                    e = float(np.max(np.abs(xn - cp) - rp))
                    worst_encl = max(worst_encl, e)
                    if e > 1e-12:
                        enclosure_ok = False
    # deterministic (a-priori) ultimate bound of Theorem 2 via Abel summation
    pdA_sig_ell = float(C.p @ C.dA @ (C.sig * ell))
    pdB_u = float(C.p @ C.dB @ np.full(D.plant.m, D.ufilt))
    eta = float(np.linalg.norm(C.Ph @ C.B @ C.K, axis=0) @ C.rbar)   # buffered-input term (Theorem 2)
    def apriori(lam):
        # V3 (corrected): the -p^T r_k term of the Abel summation cancels +p^T r_k, and the buffered-input
        # term eta <= b_K^T rbar enters every step in the worst case
        num = max(b - lam, 0) * q + c0 + pdA_sig_ell + pdB_u + C.eps0 + eta
        return num / (1 - lam)
    res = dict(bench=D.bench, conditions=conds, all_pass=all(v["pass"] for v in conds.values()),
               sampled_onestep_worst_slack=float(worst), sampled_contraction_worst=float(worst_contr),
               n_samples=n_samples, interval_enclosure_vertex_check=enclosure_ok,
               interval_enclosure_worst_excess=float(worst_encl), n_true_parameter_sets=len(true_mats),
               eta_buffer=eta, dE=D.plant.dE.tolist(), Ew=C.Ew.tolist(), lam=C.lam, eps0=C.eps0,
               N_reset_condition=dict(N_reset=20, required_gt=D.Nb + 2 * D.dmax + 2,
                                      holds=bool(20 > D.Nb + 2 * D.dmax + 2)),
               rbar=C.rbar.tolist(), ell=ell, rho=C.rho, alpha=a, beta=b, c0=c0, qbar=q, kappa=C.kap,
               sigma=C.sig.tolist(), p=C.p.tolist(), P=D.P.tolist(), KR=D.KR.tolist(),
               dA=D.plant.dA.tolist(), dB=D.plant.dB.tolist(), L=D.L, Nb=D.Nb, dmax=D.dmax, Na=D.Na,
               vbar=D.vbar, abar=D.abar, wbar=D.wbar, rel_unc=D.rel_unc, ufilt=D.ufilt, umax=D.umax,
               apriori_ultimate_bound={"lambda=rho": apriori(C.rho), "lambda=0.98": apriori(0.98)},
               apriori_note="Theorem 2 a-priori bound on ||x||_P; compare with ell (vacuous if larger).")
    return res


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "dcmotor"
    from bench_designs import make_design
    D = make_design(bench)
    r = verify(D)
    from post_alarm import grace_window
    g = grace_window(D)
    r["post_alarm_grace"] = dict(Ng=g["Ng"], W_steps=g["W_steps"], W_s=g["W_s"])
    json.dump(r, open(os.path.join(OUT, f"cert_verification_{bench}.json"), "w"), indent=1, default=float)
    print(json.dumps({k: r[k] for k in ("all_pass", "sampled_onestep_worst_slack", "sampled_contraction_worst",
                                        "interval_enclosure_vertex_check", "apriori_ultimate_bound", "ell")},
                     indent=1, default=float))
    for k, v in r["conditions"].items():
        print(k, v)
