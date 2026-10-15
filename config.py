"""Design parameters and offline synthesis for SAFE-Switch (DC motor and LFC)."""
import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are
from plant import make_dcmotor, make_lfc, make_tank
from safety import Certificate, radius_bound, design_P


def dlqr(A, B, Q, R):
    X = solve_discrete_are(A, B, Q, R)
    return -np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A), X


def dgame_are(A, B, D, Q, R, gamma, iters=5000):
    """Discrete-time zero-sum game ARE (H-infinity state feedback, full information).
    Returns K (u = K x), L (w = L x), P."""
    n = A.shape[0]
    P = Q.copy()
    BB = np.hstack([B, D])
    m, p = B.shape[1], D.shape[1]
    Rbar = np.block([[R, np.zeros((m, p))], [np.zeros((p, m)), -gamma ** 2 * np.eye(p)]])
    for _ in range(iters):
        S = Rbar + BB.T @ P @ BB
        KL = -np.linalg.solve(S, BB.T @ P @ A)
        Pn = Q + A.T @ P @ A + A.T @ P @ BB @ KL
        if np.max(np.abs(Pn - P)) < 1e-12:
            P = Pn
            break
        P = Pn
    S = Rbar + BB.T @ P @ BB
    KL = -np.linalg.solve(S, BB.T @ P @ A)
    return KL[:m], KL[m:], P


def cgame_are(A, B, D, Q, R, gamma):
    """Continuous-time GARE (8) of Huang & Dong:  A'P+PA+Q-P(BR^-1B' - gamma^-2 DD')P = 0."""
    n = A.shape[0]
    BB = np.hstack([B, D])
    Rbar = np.diag([R] + [-gamma ** 2] * D.shape[1])
    P = solve_continuous_are(A, BB, Q, Rbar)
    K = np.linalg.solve(np.atleast_2d(R), B.T @ P)  # u = -K x
    Lg = (D.T @ P) / gamma ** 2  # d = L x
    return K, Lg, P


class Design:
    """Holds every offline-synthesised quantity for one benchmark."""

    def __init__(self, bench="dcmotor", rel_unc=0.03, Ts=0.05, Nb=4, dmax=3, Na=3,
                 vbar=0.02, abar=0.03, wbar=1.0, umax=10.0, xmax=(6.0, 4.0),
                 QN=None, RN=1.0, QR=None, RR=10.0, QM=None, ufilt=5.0, eps0=0.02, lam=None, seed=0):
        self.bench = bench
        self.plant = {"dcmotor": make_dcmotor, "lfc": make_lfc, "tank": make_tank}[bench](Ts, rel_unc)
        pl = self.plant
        n = pl.n
        self.Ts, self.Nb, self.dmax, self.Na = Ts, Nb, dmax, Na
        self.L = Nb + dmax + Na
        self.vbar, self.abar, self.wbar = vbar, abar, wbar
        self.umax, self.xmax = umax, np.asarray(xmax, float)
        self.rel_unc = rel_unc
        QN = np.diag([1.0, 100.0]) if QN is None else QN
        QR = np.diag([1.0, 3.0]) if QR is None else QR
        self.KN, _ = dlqr(pl.A, pl.B, QN, RN * np.eye(pl.m))
        self.KR, _ = dlqr(pl.A, pl.B, QR, RR * np.eye(pl.m))
        QM = np.diag([1.0, 30.0]) if QM is None else QM
        self.KM, _ = dlqr(pl.A, pl.B, QM, np.eye(pl.m))  # robust-mode gain (on set estimate)
        M = pl.A + pl.B @ self.KR
        # fused half-width (vbar+abar)(1+tol) + tol*r (estimators.gate_tol = 1e-9), r <= 10*max(xmax) generously
        r_reset = np.full(n, (vbar + abar) * (1 + 1e-9) + 1e-9 * 10 * float(np.max(xmax)))

        # audit fix (V2): the filter enforces |u| <= ufilt and the radius bound uses the same ufilt
        self.ufilt = min(umax, ufilt)

        def build(P, lam_=None):
            C = Certificate(pl.A, pl.B, pl.E, pl.dA, pl.dB, wbar, self.KR, P, self.ufilt, self.xmax,
                            np.zeros(n), lam=lam_, eps0=eps0, dE=pl.dE)
            ell = C.max_level()
            rb, ok = radius_bound(C.G, pl.dA, pl.dB, pl.E, wbar, C.sig * ell, self.ufilt,
                                  r_reset, self.L, iters=300, dE=pl.dE)
            C.rbar, C.qbar = rb, float(C.p @ rb)
            return C, ell, ok

        def score(P):
            C, ell, ok = build(P)
            if not ok or C.alpha >= 1:
                return 1e9
            need = (max(C.beta - C.alpha, 0) * C.qbar + C.c0) / (1 - C.alpha)
            return max(need, C.qbar) / ell

        P, s = design_P(M, n_trials=300, seed=seed, score=score)
        self.P = P
        C, ell, ok = build(P)
        # contraction rate: allow the nominal law where possible, never below rho_R
        Ph = C.Ph
        rhoN = float(np.linalg.norm(Ph @ (pl.A + pl.B @ self.KN) @ np.linalg.inv(Ph), 2))
        lam = lam if lam is not None else float(min(max(C.rho, rhoN), 0.98))
        C.lam = lam
        C.ell = ell
        self.cert = C
        self.ell = ell
        self.cert_ok, self.cert_conditions = C.check(ell)
        self.rhoN = rhoN
        self.cert_score = s

    def summary(self):
        C = self.cert
        return dict(bench=self.bench, KN=self.KN.ravel().tolist(), KR=self.KR.ravel().tolist(), KM=self.KM.ravel().tolist(),
                    P=self.P.tolist(), ell=self.ell, rho=C.rho, rhoN=self.rhoN, lam=C.lam,
                    alpha=C.alpha, beta=C.beta, c0=C.c0, kappa=C.kap, sigma=C.sig.tolist(),
                    p=C.p.tolist(), rbar=C.rbar.tolist(), qbar=C.qbar, eps0=C.eps0,
                    margin=self.ell - ((max(C.beta - C.alpha, 0) * C.qbar + C.c0) / (1 - C.alpha)),
                    ultimate_bound_P=C.ultimate_bound(), cert_ok=self.cert_ok,
                    conditions=self.cert_conditions, dA=self.plant.dA.tolist(),
                    dB=self.plant.dB.ravel().tolist(), L=self.L, rel_unc=self.rel_unc)


if __name__ == "__main__":
    import json
    d = Design()
    print(json.dumps(d.summary(), indent=1, default=float))
