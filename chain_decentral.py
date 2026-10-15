"""E8b: compositional (assume-guarantee) certificate and decentralised SAFE-Switch for chains of
N coupled DC motors.

Each motor i has a local controller (estimator, risk model, supervisor, filter) that only uses its
own two sensor channels. The coupling sum_j A_ij x_j is treated as a bounded disturbance whose bound
|A_ij| sigma*ell follows from the neighbours' own certificates (x_j in Omega_j). If every local
certificate (C1)-(C5) holds with this augmented disturbance, Theorem 1 applies to all motors jointly
by induction over k (all subsystems are in their certified sets at k => coupling bound valid at k).
"""
import json, os, sys, time
import numpy as np
from scipy.linalg import block_diag
from config import Design
from safety import Certificate, radius_bound
from scal import chain
from controllers import SafeSwitch
from scenarios import ScenarioCfg, sample, corrupt
from train import load_risk

OUT = os.path.join(os.path.dirname(__file__), "results")


class _Plant:
    pass


class LocalDesign:
    """Design-like object for subsystem i (same interface as config.Design)."""

    def __init__(self, D1, Aii, Bii, Eii, coup_bound):
        self.__dict__.update({k: getattr(D1, k) for k in ("Ts", "Nb", "dmax", "Na", "L", "vbar", "abar",
                                                          "umax", "xmax", "rel_unc", "KN", "KR", "KM", "P", "ufilt")})
        self.bench = "dcmotor-chain"
        pl = _Plant()
        pl.A, pl.B, pl.n, pl.m = Aii, Bii, 2, 1
        pl.E = np.hstack([Eii, np.eye(2)])            # load + coupling disturbance
        pl.dA, pl.dB = D1.plant.dA, D1.plant.dB
        self.plant = pl
        self.wbar = np.r_[np.atleast_1d(D1.wbar), coup_bound]
        pl.dE = np.hstack([D1.plant.dE, np.zeros((2, 2))])
        C = Certificate(Aii, Bii, pl.E, pl.dA, pl.dB, self.wbar, D1.KR, D1.P, D1.ufilt, D1.xmax,
                        np.zeros(2), eps0=0.02, dE=pl.dE)
        ell = C.max_level()
        rb, ok = radius_bound(C.G, pl.dA, pl.dB, pl.E, self.wbar, C.sig * ell, D1.ufilt,
                              np.full(2, (D1.vbar + D1.abar) * (1 + 1e-9) + 1e-8 * float(np.max(D1.xmax))), D1.L, iters=300, dE=pl.dE)
        C.rbar, C.qbar, C.ell = rb, float(C.p @ rb), ell
        C.lam = max(0.98, C.rho)
        self.cert, self.ell = C, ell
        self.cert_ok, self.cert_conditions = C.check(ell)


def local_designs(N, kc=0.05):
    D1 = Design()
    A, B, E, _, _ = chain(N, kc=kc)
    sig_ell = D1.cert.sig * D1.ell
    out = []
    for i in range(N):
        sl = slice(2 * i, 2 * i + 2)
        coup = np.zeros(2)
        for j in range(N):
            if j != i:
                # V3: the joint discretisation also couples inputs and disturbances (B_ij, E_ij)
                coup += np.abs(A[sl, 2 * j:2 * j + 2]) @ sig_ell + np.abs(B[sl, j]) * D1.ufilt \
                    + np.abs(E[sl, j]) * D1.wbar
        out.append(LocalDesign(D1, A[sl, sl], B[sl, i:i + 1], E[sl, i:i + 1], coup))
    return out, (A, B, E), D1


def certificate_table(Ns=(1, 2, 4, 8, 16, 32), kc=0.05):
    rows = []
    for N in Ns:
        Ls, _, _ = local_designs(N, kc)
        worst = min(Ls, key=lambda d: d.ell - (d.cert.alpha * d.ell + max(d.cert.beta - d.cert.alpha, 0) *
                                               d.cert.qbar + d.cert.c0))
        C = worst.cert
        lhs = C.alpha * worst.ell + max(C.beta - C.alpha, 0) * C.qbar + C.c0
        rows.append(dict(N=N, kc=kc, all_local_ok=all(d.cert_ok for d in Ls), alpha=C.alpha, c0=C.c0,
                         qbar=C.qbar, ell=worst.ell, C2_lhs=lhs, C2_margin=worst.ell - lhs,
                         coupling_bound=worst.wbar[1:].tolist()))
    return rows


def simulate_chain(N, seed, scen, T=600, sup="threshold", kc=0.05):
    Ls, (A, B, E), D1 = local_designs(N, kc)
    risk = load_risk()
    rng = np.random.default_rng(seed)
    # true plant: nominal chain + time-invariant element-wise model error within the local bounds
    dA = block_diag(*[D1.plant.dA * rng.uniform(-1, 1, (2, 2)) for _ in range(N)])
    dB = block_diag(*[D1.plant.dB * rng.uniform(-1, 1, (2, 1)) for _ in range(N)])
    At, Bt = A + dA, B + dB
    scs = [sample(ScenarioCfg(name="chain", **scen, attack_channels=((seed + i) % 2,)), 50_000 + 97 * seed + i)
           for i in range(N)]
    ctrls = [SafeSwitch(Ls[i], sup, risk=risk) for i in range(N)]
    x = np.concatenate([sc["x0"] for sc in scs]) * 0.5
    for i in range(N):
        ctrls[i].reset(scs[i], x[2 * i:2 * i + 2])
    arrivals = [[[] for _ in range(T + 8)] for _ in range(N)]
    u_app = np.zeros(N); bufs = [[] for _ in range(N)]
    viol = 0; cont_fail = 0; infeas = 0; tloc = []
    err = []
    for k in range(T):
        U = np.zeros(N)
        for i in range(N):
            xi = x[2 * i:2 * i + 2]
            y, _ = corrupt(scs[i], k, xi, scs[i]["v"][k])
            for ch in range(2):
                if not scs[i]["s_lost"][k, ch]:
                    arrivals[i][k + int(scs[i]["s_delay"][k, ch])].append((ch, k, float(y[ch])))
            t0 = time.perf_counter()
            u, buf, info = ctrls[i].step(k, arrivals[i][k], np.array([u_app[i]]))
            tloc.append(time.perf_counter() - t0)
            infeas += (not info["feas"])
            cont_fail += int(not np.all(np.abs(xi - info["c"]) <= info["r"] + 1e-9))
            if not scs[i]["a_lost"][k]:
                U[i] = float(np.clip(u[0], -D1.ufilt, D1.ufilt)); bufs[i] = list(buf) if buf else []
            else:
                U[i] = float(np.clip(bufs[i].pop(0)[0], -D1.ufilt, D1.ufilt)) if bufs[i] else u_app[i]
        u_app = U
        w = np.array([sc["w"][k, 0] for sc in scs])
        x = At @ x + Bt @ u_app + E @ w
        viol += int(np.any(np.abs(x.reshape(N, 2)) > D1.xmax))
        err.append(np.linalg.norm(x) / np.sqrt(N))
    return dict(N=N, seed=seed, viol_steps=viol, viol_any=float(viol > 0), contain=1 - cont_fail / (T * N),
                infeas=infeas / (T * N), rmse=float(np.sqrt(np.mean(np.square(err)))),
                t_local_ms=float(np.mean(tloc) * 1e3), t_local_p99_ms=float(np.percentile(tloc, 99) * 1e3),
                t_total_ms=float(np.mean(tloc) * 1e3 * N))


if __name__ == "__main__":
    rows = certificate_table()
    for r in rows:
        print(r)
    env = dict(p_loss=0.1, dmax=3, pa_loss=0.1)           # inside the envelope
    att = dict(attack="bias", attack_mag=6.0, on_mean=0.1, min_on=0.1, off_mean=1.0, **env)  # E10 bursts
    sims = []
    for N in [2, 4, 8, 16]:
        for s in range(5):
            for nm, sc in [("network", env), ("burst-bias+network", att)]:
                r = simulate_chain(N, s, sc)
                r["scenario"] = nm
                sims.append(r)
                print(r, flush=True)
    json.dump(dict(certificates=rows, simulations=sims), open(os.path.join(OUT, "chain_decentral.json"), "w"),
              indent=1, default=float)
