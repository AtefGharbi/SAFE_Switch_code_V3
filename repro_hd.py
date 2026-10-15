"""Reproduction of the DC-motor experiment of Huang & Dong (2020), Section IV.

Plant: Table II parameters, x0 = [1, 1], B_d = [0.1, 0.3]^T, white disturbance (power 0.1).
Fault: x~ = Lam x, Lam = diag(-1.5, -0.5), intermittent (fault active >= 60 % of the time).
Controllers: (a) H-infinity pair of mode 0 only (S-ADPC-like non-switched controller),
             (b) switched H-infinity controller with prescribed-performance switching law (33)
                 (eta0 = 1.1, eta_inf = 0.05, a0 = 2.6, a1 = 1.7).
Simulation step 0.01 s (sampled-data approximation of the continuous-time design), 800 s.
"""
import json, os
import numpy as np
from plant import make_dcmotor
from config import cgame_are
from controllers import HuangDong2020

OUT = os.path.join(os.path.dirname(__file__), "results")


class _D:  # light design stub for the baseline class
    def __init__(self, pl):
        self.plant = pl
        self.umax = 1e3


def fault_schedule(rng, T, Ts, on_mean=30.0, off_mean=12.0):
    act = np.zeros(T, bool)
    t = 0.0
    while t < T * Ts:
        t += rng.exponential(off_mean)
        d = rng.exponential(on_mean)
        act[int(t / Ts):int(min(T * Ts, t + d) / Ts)] = True
        t += d
    return act


def run(seed=0, Tsec=800.0, Ts=0.01, switched=True, eta_inf=0.05):
    pl = make_dcmotor(Ts=Ts, rel_unc=0.0)
    rng = np.random.default_rng(seed)
    T = int(Tsec / Ts)
    act = fault_schedule(rng, T, Ts)
    Lam = np.diag([-1.5, -0.5])
    hd = HuangDong2020(_D(pl), eta_inf=eta_inf)
    x = np.array([1.0, 1.0])
    hd.reset(None, x.copy())
    w = rng.normal(0, np.sqrt(0.1), T)
    nx = np.zeros(T); sig = np.zeros(T, int); sigp = np.zeros(T, int); V = np.zeros(T)
    for k in range(T):
        xt = Lam @ x if act[k] else x.copy()
        if switched:
            u, _, info = hd.step(k, [(0, k, xt[0]), (1, k, xt[1])], None)
            sigp[k] = hd.mode
        else:
            u = hd.K[0] @ xt
            sigp[k] = 0
        sig[k] = int(act[k])
        V[k] = float(xt @ hd.P[sigp[k]] @ xt)
        nx[k] = np.linalg.norm(x)
        x = pl.A @ x + pl.B @ np.atleast_1d(u) + pl.E[:, 0] * w[k]
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e12:
            nx[k + 1:] = np.nan
            break
    return dict(t=np.arange(T) * Ts, nx=nx, sig=sig, sigp=sigp, V=V, act=act)


def gains():
    pl = make_dcmotor(Ts=0.01, rel_unc=0.0)
    K0, L0, P0 = cgame_are(pl.Ac, pl.Bc, pl.Dc, np.eye(2), 1.0, 2.0)
    Lam = np.diag([-1.5, -0.5])
    At = Lam @ pl.Ac @ np.linalg.inv(Lam)
    K1, L1, P1 = cgame_are(At, Lam @ pl.Bc, Lam @ pl.Dc, np.eye(2), 1.0, 2.0)
    # Huang & Dong report W_{2,0} = [0.1557 0.2988 0.2776] with psi_2 = [x1^2, x1x2, x2^2],
    # i.e. u = -R^-1 B^T P x with P = [[W1, W2/2], [W2/2, W3]] -> K0 = 6.6797*[0.1557, 0.1494]
    W20 = np.array([0.1557, 0.2988, 0.2776])
    K0_paper = 6.6797 * np.array([W20[0], W20[1] / 2])
    W21 = np.array([0.0713, 0.1986, 0.5630])
    K1_paper = -10.0195 * np.array([W21[0], W21[1] / 2])  # B_u,1 = [-10.0195, 0]
    return dict(A=pl.Ac.tolist(), B=pl.Bc.ravel().tolist(), K0=K0.ravel().tolist(), K1=K1.ravel().tolist(),
                P0=P0.tolist(), P1=P1.tolist(), K0_paper=K0_paper.tolist(), K1_paper=K1_paper.tolist(),
                Ahat0_paper=[[-6.900, -0.0209], [7.4580, 0.0]], Bhat0_paper=[6.6797, 0.0])


if __name__ == "__main__":
    g = gains()
    res = {"gains": g}
    sw = run(0, switched=True)
    ns = run(0, switched=False)
    res["switched"] = dict(max_norm=float(np.nanmax(sw["nx"])), final_norm=float(sw["nx"][-1]),
                           n_switch=int(np.sum(np.abs(np.diff(sw["sigp"])))),
                           fault_frac=float(sw["act"].mean()),
                           mismatch_frac=float(np.mean(sw["sig"] != sw["sigp"])))
    ns_div = np.where(~np.isfinite(ns["nx"]) | (ns["nx"] > 1e6))[0]
    res["nonswitched"] = dict(max_norm=float(np.nanmax(ns["nx"])),
                              diverged=bool(len(ns_div) > 0 or np.nanmax(ns["nx"]) > 1e3),
                              t_exceed_1e3=float(ns["t"][np.argmax(ns["nx"] > 1e3)]) if np.nanmax(ns["nx"]) > 1e3 else None)
    # sensitivity to eta_inf (their Fig. 8): count of chattering spikes
    res["eta_sweep"] = {}
    for e in [50, 5, 0.5, 0.05]:
        r = run(0, switched=True, eta_inf=e)
        res["eta_sweep"][str(e)] = dict(max_norm=float(np.nanmax(r["nx"])),
                                        mean_norm=float(np.nanmean(r["nx"])),
                                        n_switch=int(np.sum(np.abs(np.diff(r["sigp"])))))
    json.dump(res, open(os.path.join(OUT, "repro_hd.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "repro_hd_traces.npz"), sw_t=sw["t"][::10], sw_nx=sw["nx"][::10],
                        sw_sig=sw["sig"][::10], sw_sigp=sw["sigp"][::10], ns_nx=ns["nx"][::10],
                        ns_t=ns["t"][::10])
    print(json.dumps(res, indent=1))
