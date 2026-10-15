"""Independent re-computation of every certificate quantity (V3 audit).

Re-derives, WITHOUT using safety.Certificate, config.Design or radius_bound:
discretisation (matrix exponential), the uncertainty bounds dA, dB, dE (parameter gridding), the fallback gain
K_R (discrete Riccati equation), p, sigma, kappa, rho (generalised eigenvalue route), alpha, beta, c0, ell,
the radius fixed point rbar of (8), qbar and conditions (C1)-(C5); then compares each number with the
machine-readable file results/cert_verification_<bench>.json written by verify_certificate.py.
Only the physical continuous-time model (plant.dcmotor_ct / plant.tank_ct) and the design choice P are reused.
"""
import itertools, json, os, sys
import numpy as np
from scipy.linalg import expm, solve_discrete_are, eigh

OUT = os.path.join(os.path.dirname(__file__), "results")

SPEC = {
    "dcmotor": dict(ct="dcmotor_ct", nominal="DCM_NOMINAL", keys=["RT", "CT", "J", "LT"], Ts=0.05, rel=0.03,
                    QR=np.diag([1.0, 3.0]), RR=10.0, wbar=1.0, xmax=np.array([6.0, 4.0]), ubar=5.0,
                    vbar=0.02, abar=0.03, Nb=4, dmax=3, Na=3),
    "tank": dict(ct="tank_ct", nominal="TANK_NOMINAL", keys=["A1", "A2", "a1", "a2", "Kp"], Ts=0.5, rel=0.03,
                 QR=np.diag([1.0, 3.0]), RR=10.0, wbar=2.0, xmax=np.array([6.0, 6.0]), ubar=5.0,
                 vbar=0.02, abar=0.03, Nb=4, dmax=3, Na=3),
}


def zoh(Ac, Bc, Dc, Ts):
    n, m, p = Ac.shape[0], Bc.shape[1], Dc.shape[1]
    Z = np.zeros((n + m + p, n + m + p)); Z[:n, :n] = Ac; Z[:n, n:n + m] = Bc; Z[:n, n + m:] = Dc
    F = expm(Z * Ts)
    return F[:n, :n], F[:n, n:n + m], F[:n, n + m:]


def audit(bench):
    import plant
    sp = SPEC[bench]
    ct = getattr(plant, sp["ct"]); nom = getattr(plant, sp["nominal"])
    A, B, E = zoh(*ct(**nom), sp["Ts"])
    dA, dB, dE = np.zeros_like(A), np.zeros_like(B), np.zeros_like(E)
    for s in itertools.product(np.linspace(1 - sp["rel"], 1 + sp["rel"], 5), repeat=len(sp["keys"])):
        pp = dict(nom); pp.update({k: nom[k] * f for k, f in zip(sp["keys"], s)})
        At, Bt, Et = zoh(*ct(**pp), sp["Ts"])
        dA, dB, dE = np.maximum(dA, abs(At - A)), np.maximum(dB, abs(Bt - B)), np.maximum(dE, abs(Et - E))
    dA, dB, dE = 1.02 * dA, 1.02 * dB, 1.02 * dE
    # rigour of the gridding: a finer 9-point grid plus random interior points, WITHOUT margin, must stay below
    fA, fB, fE = np.zeros_like(A), np.zeros_like(B), np.zeros_like(E)
    rng0 = np.random.default_rng(7)
    grid9 = list(itertools.product(np.linspace(1 - sp["rel"], 1 + sp["rel"], 9), repeat=len(sp["keys"])))
    rnd = [tuple(1 + sp["rel"] * rng0.uniform(-1, 1, len(sp["keys"]))) for _ in range(3000)]
    for s in grid9 + rnd:
        pp = dict(nom); pp.update({k: nom[k] * f for k, f in zip(sp["keys"], s)})
        At, Bt, Et = zoh(*ct(**pp), sp["Ts"])
        fA, fB, fE = np.maximum(fA, abs(At - A)), np.maximum(fB, abs(Bt - B)), np.maximum(fE, abs(Et - E))
    fine_ok = bool(np.all(fA <= dA + 1e-15) and np.all(fB <= dB + 1e-15) and np.all(fE <= dE + 1e-15))
    fine_ratio = float(max(np.max(np.where(dA > 0, fA / np.maximum(dA, 1e-300), 0)),
                           np.max(np.where(dB > 0, fB / np.maximum(dB, 1e-300), 0)),
                           np.max(np.where(dE > 0, fE / np.maximum(dE, 1e-300), 0))))
    X = solve_discrete_are(A, B, sp["QR"], sp["RR"] * np.eye(1))
    KR = -np.linalg.solve(sp["RR"] * np.eye(1) + B.T @ X @ B, B.T @ X @ A)
    rep = json.load(open(os.path.join(OUT, f"cert_verification_{bench}.json")))
    P = np.array(rep["P"])                                   # design choice (searched), reused
    M = A + B @ KR
    Pinv = np.linalg.inv(P)
    p = np.sqrt(np.diag(P)); sig = np.sqrt(np.diag(Pinv))
    kap = float(np.sqrt(np.max(np.diag(KR @ Pinv @ KR.T))))
    rho = float(np.sqrt(eigh(M.T @ P @ M, P, eigvals_only=True).max()))   # ||P^1/2 M P^-1/2||_2
    G = abs(A) + dA
    wbar = np.atleast_1d(sp["wbar"])
    Ew = (abs(E) + dE) @ wbar
    alpha = rho + p @ dA @ sig + float(p @ dB.sum(1)) * kap
    beta = float(np.max((p @ G) / p))
    c0 = float(p @ Ew)
    ell = float(min(sp["ubar"] / kap, np.min(sp["xmax"] / sig)))
    L = sp["Nb"] + sp["dmax"] + sp["Na"]
    g = dA @ (sig * ell) + dB.sum(1) * sp["ubar"] + Ew
    r0 = np.full(2, (sp["vbar"] + sp["abar"]) * (1 + 1e-9) + 1e-8 * float(np.max(sp["xmax"])))
    R = r0.copy()
    for _ in range(5000):                                    # least fixed point of (8), Kleene iteration
        Rn = np.empty_like(R)
        for i in range(2):
            phi, best = r0[i], r0[i]
            for _j in range(L):
                phi = G[i, i] * phi + sum(G[i, l] * R[l] for l in range(2) if l != i) + g[i]
                best = max(best, phi)
            Rn[i] = best
        if np.max(abs(Rn - R)) < 1e-14:
            R = Rn; break
        R = Rn
    qbar = float(p @ R)
    lam = rep["lam"]
    C2 = alpha * ell + max(beta - alpha, 0) * qbar + c0
    mine = dict(rho=rho, alpha=alpha, beta=beta, c0=c0, kappa=kap, ell=ell, qbar=qbar,
                rbar=R.tolist(), sigma=sig.tolist(), p=p.tolist(), KR=KR.tolist(), dA=dA.tolist(), dB=dB.tolist(),
                dE=dE.tolist(), C2_lhs=C2, C2_margin=ell - C2, C1=alpha < 1, C3=qbar < ell,
                C4=bool(kap * ell <= sp["ubar"] + 1e-12 and np.all(sig * ell <= sp["xmax"] + 1e-12)), C5=lam >= rho)
    theirs = dict(rho=rep["rho"], alpha=rep["alpha"], beta=rep["beta"], c0=rep["c0"], kappa=rep["kappa"],
                  ell=rep["ell"], qbar=rep["qbar"], rbar=rep["rbar"], sigma=rep["sigma"], p=rep["p"], KR=rep["KR"],
                  dA=rep["dA"], dB=rep["dB"], dE=rep["dE"], C2_lhs=rep["conditions"]["C2_level_invariance"]["lhs"],
                  C2_margin=rep["conditions"]["C2_level_invariance"]["margin"],
                  C1=rep["conditions"]["C1_alpha_lt_1"]["pass"], C3=rep["conditions"]["C3_qbar_lt_ell"]["pass"],
                  C4=rep["conditions"]["C4a_input"]["pass"] and rep["conditions"]["C4b_state"]["pass"],
                  C5=rep["conditions"]["C5_lambda_ge_rho"]["pass"])
    cmp = {}
    for k in mine:
        a, b = np.asarray(mine[k], float), np.asarray(theirs[k], float)
        d = float(np.max(np.abs(a - b)))
        cmp[k] = dict(independent=mine[k], reported=theirs[k], max_abs_diff=d,
                      match=bool(d <= 1e-9 * max(1.0, float(np.max(np.abs(b))))))
    # independent sampled check of the key inequality on I (own prediction operator)
    rng = np.random.default_rng(123)
    Ph = np.real(np.linalg.cholesky(P)).T                    # ||x||_P = ||Ph x|| with P = Ph^T Ph
    nP = lambda v: float(np.linalg.norm(Ph @ v))
    worst = -np.inf
    for _ in range(100_000):
        r = rng.uniform(0, 1, 2) ** 0.3 * R
        smax = max(ell - p @ r, 0.0)
        d = rng.normal(size=2); d /= nP(d)
        c = d * smax * rng.uniform(0.9, 1.0)
        u = KR @ c
        cn = A @ c + B @ u
        rn = G @ r + dA @ abs(c) + dB @ abs(u) + Ew
        worst = max(worst, nP(cn) + p @ rn - ell)
    out = dict(bench=bench, comparisons=cmp, all_match=all(v["match"] for v in cmp.values()),
               independent_sampled_key_inequality_worst_slack=worst, n_samples=100_000,
               fine_grid_uncertainty_check=dict(ok=fine_ok, max_ratio_to_reported=fine_ratio,
                                                n_points=len(grid9) + len(rnd)),
               note="P is the searched design matrix (reused); everything else recomputed independently.")
    json.dump(out, open(os.path.join(OUT, f"cert_audit_{bench}.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    for b in (sys.argv[1:] or ["dcmotor", "tank"]):
        o = audit(b)
        bad = [k for k, v in o["comparisons"].items() if not v["match"]]
        print(b, "fine-grid delta check", o["fine_grid_uncertainty_check"])
        print(b, "all_match", o["all_match"], "mismatches", bad, "sampled worst slack",
              round(o["independent_sampled_key_inequality_worst_slack"], 4))
        for k in ("rho", "alpha", "beta", "c0", "ell", "qbar", "C2_margin"):
            print("   ", k, o["comparisons"][k]["independent"], o["comparisons"][k]["max_abs_diff"])
