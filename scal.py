"""E8: scalability of the SAFE-Switch runtime components on chains of coupled DC motors.

n = 2N states, m = N inputs, p = N load disturbances. Measured per control step:
 - set estimator (hybrid zonotope/box, one time update + n timestamped packets, delay <= 3)
 - Kalman filter with OOSM reprocessing
 - safety filter (exact 1-D projection for m = 1; parametrised SOCP (Clarabel) for m > 1)
 - PPO supervisor forward pass
 plus offline synthesis time and whether the Theorem-2 certificate conditions hold.
"""
import json, os, time
import numpy as np
from scipy.linalg import block_diag, solve_discrete_lyapunov
from plant import c2d, dcmotor_ct
from config import dlqr
from safety import Certificate, radius_bound
from estimators import IntervalEstimator, KalmanOOSM
from ppo import PPOPolicy

OUT = os.path.join(os.path.dirname(__file__), "results")


def chain(N, kc=0.05, Ts=0.05, rel=0.03):
    A1, B1, D1 = dcmotor_ct()
    Ac = block_diag(*[A1] * N)
    for i in range(N - 1):
        a, b = 2 * i + 1, 2 * (i + 1) + 1
        Ac[a, a] -= kc; Ac[a, b] += kc; Ac[b, b] -= kc; Ac[b, a] += kc
    Bc = block_diag(*[B1] * N)
    Dc = block_diag(*[D1] * N)
    A, B, E = c2d(Ac, Bc, Dc, Ts)
    # element-wise uncertainty bounds scaled from the single-motor analysis (3 %)
    from plant import make_dcmotor
    pl1 = make_dcmotor(Ts, rel)
    dA = block_diag(*[pl1.dA] * N)
    dB = block_diag(*[pl1.dB] * N)
    return A, B, E, dA, dB


class ParamFilter:
    def __init__(self, C):
        import cvxpy as cp
        n, m = C.B.shape
        self.C = C
        self.u = cp.Variable(m)
        self.uc = cp.Parameter(m)
        self.c = cp.Parameter(n)
        self.k1 = cp.Parameter()      # ell - p'(G r + dA|c| + |E|w)
        self.k2 = cp.Parameter()      # lam ||c||_P + eps0
        cpred = C.A @ self.c + C.B @ self.u
        pdB = C.p @ C.dB
        cons = [cp.norm(C.Ph @ cpred) + pdB @ cp.abs(self.u) <= self.k1,
                cp.norm(C.Ph @ cpred) <= self.k2, cp.abs(self.u) <= C.umax]
        self.prob = cp.Problem(cp.Minimize(cp.sum_squares(self.u - self.uc)), cons)

    def __call__(self, c, r, uc):
        C = self.C
        self.uc.value = uc
        self.c.value = c
        self.k1.value = C.ell - float(C.p @ (C.G @ r + C.dA @ np.abs(c) + C.Ew))
        self.k2.value = C.lam * C.normP(c) + C.eps0
        self.prob.solve(solver="CLARABEL")
        return self.u.value


def bench(N, steps=150, seed=0):
    rng = np.random.default_rng(seed)
    t_syn = time.perf_counter()
    A, B, E, dA, dB = chain(N)
    n, m = B.shape
    from config import Design as _Dz
    _d1 = _Dz()
    KR = block_diag(*[_d1.KR] * N)  # decentralised certified fallback gain
    KN, _ = dlqr(A, B, block_diag(*[np.diag([1.0, 100.0])] * N), np.eye(m))
    from config import Design
    P = block_diag(*[Design().P] * N)  # block-diagonal certificate matrix from the single-motor design
    xmax = np.tile([6.0, 4.0], N)
    C = Certificate(A, B, E, dA, dB, np.ones(m), KR, P, 5.0, xmax, np.zeros(n), eps0=0.02)
    ell = C.max_level()
    rb, ok = radius_bound(C.G, dA, dB, E, np.ones(m), C.sig * ell, 5.0, np.full(n, 0.05), 10, iters=300)
    C.rbar, C.qbar, C.ell, C.lam = rb, float(C.p @ rb), ell, 0.98
    cert_ok, _ = C.check(ell)
    t_syn = time.perf_counter() - t_syn
    est = IntervalEstimator(C, np.full(n, 0.05), 14, np.zeros(n), np.full(n, 0.05), max_gen=max(8, 2 * n))
    kf = KalmanOOSM(A, B, E, 0.2, 1e-4, np.zeros(n), np.eye(n) * 1e-4, 14)
    pol = PPOPolicy(12, seed=0)
    filt = ParamFilter(C) if m > 1 else None
    x = rng.uniform(-0.3, 0.3, n)
    te, tk, tf, tp = [], [], [], []
    u = np.zeros(m)
    for k in range(1, steps):
        y = x + rng.uniform(-0.02, 0.02, n)
        t0 = time.perf_counter()
        est.time_update(k, u)
        est.measure_batch([(i, max(0, k - int(rng.integers(0, 4))), y[i]) for i in range(n)])  # batched update
        te.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        kf.time_update(k, u)
        kf.measure_batch([(i, max(0, k - int(rng.integers(0, 4))), y[i]) for i in range(n)])
        tk.append(time.perf_counter() - t0)
        c, r = est.box
        t0 = time.perf_counter()
        pol.act(rng.normal(size=12), deterministic=True)
        tp.append(time.perf_counter() - t0)
        uc = KN @ kf.x
        t0 = time.perf_counter()
        if m == 1:
            us, _, _ = C.filter(c, r, uc)
        else:
            try:
                us = filt(c, r, uc)
            except Exception:
                us = None
            if us is None:
                us = KR @ c
        tf.append(time.perf_counter() - t0)
        u = np.clip(np.asarray(us).ravel(), -10, 10)
        x = A @ x + B @ u + E @ rng.uniform(-0.3, 0.3, m)
        x = np.clip(x, -10, 10)
    f = lambda a: (float(np.mean(a) * 1e3), float(np.percentile(a, 99) * 1e3))
    return dict(N=N, n=n, m=m, est_ms=f(te), kf_ms=f(tk), filter_ms=f(tf), ppo_ms=f(tp),
                total_ms=float((np.mean(te) + np.mean(tk) + np.mean(tf) + np.mean(tp)) * 1e3),
                synthesis_s=t_syn, cert_ok=bool(cert_ok), alpha=C.alpha, qbar_over_ell=C.qbar / ell)


if __name__ == "__main__":
    res = [bench(N, steps=100) for N in [1, 2, 4, 8, 16]]
    json.dump(res, open(os.path.join(OUT, "scal.json"), "w"), indent=1)
    for r in res:
        print(r)
