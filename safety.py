"""Information-state safety certificate and robust safety filter of SAFE-Switch.

Information state: box  B(c, r) = {x : |x - c| <= r}  (element-wise), with c the
set-estimator centre and r the radius produced by the interval set estimator.

Certified set (ellipsoidal level set of P):
      I = {(c, r) :  f(c, r) := ||c||_P + p^T r <= ell ,  r <= rbar }
with p_i = ||P^{1/2} e_i||. f(c, r) <= ell implies B(c, r) is contained in
Omega = {x : ||x||_P <= ell} (triangle inequality), and Omega ⊆ X_safe.

One-step robust prediction for input u (all A in A_hat ± dA, B in B_hat ± dB, |w| <= wbar):
      c+ = A_hat c + B_hat u
      r+ = G r + dA |c| + dB |u| + |E| wbar,   G = |A_hat| + dA

Safety filter (for each step k):
      u_safe = argmin_u ||u - u_cand||^2
      s.t.   f(c+(u), r+(u)) <= ell                       (robust containment)
             ||c+(u)||_P <= lam ||c||_P + eps0            (centre contraction)
             |u| <= umax.
For single-input plants the feasible set is an interval (convex constraints in a
scalar) and the projection is computed exactly by 1-D convex root finding; for
m > 1 the same problem is a second-order-cone program (solved with CVXPY).
"""
import numpy as np
from scipy.linalg import sqrtm, solve_discrete_lyapunov


class Certificate:
    def __init__(self, A, B, E, dA, dB, wbar, K, P, umax, xmax, rbar, lam=None, eps0=0.0, dE=None):
        self.A, self.B, self.E = A, B, E
        self.dA, self.dB = dA, dB
        self.wbar = np.atleast_1d(wbar).astype(float)
        self.dE = np.zeros_like(E) if dE is None else np.asarray(dE, float)
        self.Ew = (np.abs(E) + self.dE) @ self.wbar      # bound on |E_true w| (V3)
        self.K, self.P = K, P
        self.umax, self.xmax = umax, np.asarray(xmax, float)
        self.Ph = np.real(sqrtm(P))
        self.Pinv = np.linalg.inv(P)
        self.p = np.sqrt(np.diag(P))
        self.sig = np.sqrt(np.diag(self.Pinv))
        self.kap = float(np.sqrt(np.max(np.diag(K @ self.Pinv @ K.T))))   # max_j sqrt(K_j P^-1 K_j^T)
        M = A + B @ K
        Phi = np.linalg.inv(self.Ph)
        self.rho = float(np.linalg.norm(self.Ph @ M @ Phi, 2))
        self.G = np.abs(A) + dA
        self.alpha = self.rho + self.p @ dA @ self.sig + float(self.p @ dB @ np.ones(B.shape[1])) * self.kap
        self.beta = float(np.max((self.p @ self.G) / self.p))
        self.c0 = float(self.p @ self.Ew)
        self.rbar = np.asarray(rbar, float)
        self.qbar = float(self.p @ self.rbar)
        self.lam = self.rho if lam is None else lam
        self.eps0 = eps0
        self.ell = None

    # --- certificate --------------------------------------------------------
    def max_level(self):
        """Largest ell allowed by input and state constraints."""
        l_u = self.umax / self.kap
        l_x = np.min(self.xmax / self.sig)
        return min(l_u, l_x)

    def check(self, ell):
        """Conditions of Theorem 2 (recursive feasibility)."""
        a, b, c0, q = self.alpha, self.beta, self.c0, self.qbar
        cond = {
            "alpha<1": a < 1,
            "contraction": (a * ell + max(b - a, 0.0) * q + c0 <= ell) and (a * ell + c0 <= ell),
            "nonempty": q < ell,
            "input": self.kap * ell <= self.umax + 1e-12,
            "state": bool(np.all(self.sig * ell <= self.xmax + 1e-12)),
            "lam>=rho": self.lam >= self.rho - 1e-12,
        }
        return all(cond.values()), cond

    def ultimate_bound(self):
        """Theorem 3: limsup ||x||_P <= (eps0 + qbar)/(1-lam) + qbar."""
        return (self.eps0 + self.qbar) / (1 - self.lam) + self.qbar

    # --- information-state functions ---------------------------------------
    def normP(self, x):
        return float(np.linalg.norm(self.Ph @ x))

    def f(self, c, r):
        return self.normP(c) + float(self.p @ r)

    def predict(self, c, r, u):
        u = np.atleast_1d(u)
        cp = self.A @ c + self.B @ u
        rp = self.G @ r + self.dA @ np.abs(c) + self.dB @ np.abs(u) + self.Ew
        return cp, rp

    def g_contain(self, c, r, u):
        cp, rp = self.predict(c, r, u)
        return self.normP(cp) + float(self.p @ rp) - self.ell

    def g_contract(self, c, u):
        cp = self.A @ c + self.B @ np.atleast_1d(u)
        return self.normP(cp) - (self.lam * self.normP(c) + self.eps0)

    # --- filter ---------------------------------------------------------------
    def filter(self, c, r, u_cand):
        """Return (u_safe, feasible_flag, intervened_flag)."""
        if self.B.shape[1] == 1:
            return self._filter_1d(c, r, float(np.atleast_1d(u_cand)[0]))
        return self._filter_nd(c, r, np.atleast_1d(u_cand))

    def _g(self, c, r, u):
        return max(self.g_contain(c, r, u), self.g_contract(c, u))

    def _filter_1d(self, c, r, uc):
        um = self.umax
        uc_cl = min(max(uc, -um), um)
        if self._g(c, r, uc_cl) <= 0:
            return np.array([uc_cl]), True, abs(uc_cl - uc) > 1e-9
        # g is convex in u: locate its minimiser by golden-section search on [-um, um]
        lo, hi = -um, um
        gr = (np.sqrt(5) - 1) / 2
        a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
        ga, gb = self._g(c, r, a), self._g(c, r, b)
        for _ in range(40):
            if ga < gb:
                hi, b, gb = b, a, ga
                a = hi - gr * (hi - lo)
                ga = self._g(c, r, a)
            else:
                lo, a, ga = a, b, gb
                b = lo + gr * (hi - lo)
                gb = self._g(c, r, b)
        ustar = 0.5 * (lo + hi)
        if self._g(c, r, ustar) > 0:
            # infeasible: verified fallback = certified robust law, saturated
            uf = float(np.clip(self.K @ c, -um, um)[0])
            return np.array([uf]), False, True
        # bisection between feasible ustar and infeasible uc_cl for the boundary
        a, b = ustar, uc_cl
        for _ in range(40):
            mid = 0.5 * (a + b)
            if self._g(c, r, mid) <= 0:
                a = mid
            else:
                b = mid
        return np.array([a]), True, True

    def _filter_nd(self, c, r, uc):
        import cvxpy as cp
        m = self.B.shape[1]
        u = cp.Variable(m)
        cp_ = self.A @ c + self.B @ u
        rp = self.G @ r + self.dA @ np.abs(c) + self.dB @ cp.abs(u) + self.Ew
        cons = [cp.norm(self.Ph @ cp_) + self.p @ rp <= self.ell,
                cp.norm(self.Ph @ cp_) <= self.lam * self.normP(c) + self.eps0,
                cp.abs(u) <= self.umax]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(u - uc)), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
            ok = prob.status in ("optimal", "optimal_inaccurate")
        except Exception:
            ok = False
        if not ok:
            return np.clip(self.K @ c, -self.umax, self.umax), False, True
        us = np.asarray(u.value).ravel()
        return us, True, bool(np.linalg.norm(us - uc) > 1e-6)


def radius_bound(G, dA, dB, E, wbar, xabs_max, umax, r_reset, L, iters=500, dE=None):
    """Uniform bound rbar on the estimator radius (Lemma 1).

    Each channel i is reset to <= r_reset_i at least once every L steps; between
    resets component i obeys r_i+ <= G_ii r_i + sum_{j!=i} G_ij R_j + g_i with all
    other components bounded by R (induction hypothesis). The least fixed point of
    this monotone map is a valid bound.
    """
    Eb = np.abs(E) + (0 if dE is None else dE)
    g = dA @ xabs_max + dB @ (umax * np.ones(dB.shape[1])) + Eb @ np.atleast_1d(wbar)
    n = G.shape[0]
    R = np.array(r_reset, float).copy()
    for _ in range(iters):
        Rn = R.copy()
        for i in range(n):
            ri = r_reset[i]
            best = ri
            for _j in range(L):
                ri = G[i, i] * ri + sum(G[i, j] * R[j] for j in range(n) if j != i) + g[i]
                best = max(best, ri)
            Rn[i] = max(R[i], best)
        if np.allclose(Rn, R, rtol=0, atol=1e-12):
            return Rn, True
        R = Rn
        if np.any(R > 1e6):
            return R, False
    return R, False


def design_P(M, n_trials=4000, seed=0, score=None):
    """Search Lyapunov matrices P (normalised det) minimising a user score."""
    rng = np.random.default_rng(seed)
    best, bestP = np.inf, None
    n = M.shape[0]
    cands = [solve_discrete_lyapunov(M.T, np.eye(n))]
    for _ in range(n_trials):
        L = np.tril(rng.normal(size=(n, n)))
        Qm = L @ L.T + 1e-3 * np.eye(n)
        cands.append(solve_discrete_lyapunov(M.T, Qm))
    for P in cands:
        P = P / np.linalg.det(P) ** (1 / n)
        s = score(P)
        if s < best:
            best, bestP = s, P
    return bestP, best


# ---------------------------------------------------------------- exact 1-D projection
def _quad_le0(a2, a1, a0, lo, hi, tol=1e-12):
    """Intervals of {u in [lo,hi] : a2 u^2 + a1 u + a0 <= 0}."""
    if lo > hi:
        return []
    if abs(a2) < tol:
        if abs(a1) < tol:
            return [(lo, hi)] if a0 <= 0 else []
        r = -a0 / a1
        return [(lo, min(hi, r))] if a1 > 0 and lo <= min(hi, r) else \
               ([(max(lo, r), hi)] if a1 < 0 and max(lo, r) <= hi else [])
    disc = a1 * a1 - 4 * a2 * a0
    if disc < 0:
        return [(lo, hi)] if a2 < 0 else []
    s = np.sqrt(disc)
    r1, r2 = sorted(((-a1 - s) / (2 * a2), (-a1 + s) / (2 * a2)))
    if a2 > 0:
        L, H = max(lo, r1), min(hi, r2)
        return [(L, H)] if L <= H else []
    out = []
    if lo <= min(hi, r1):
        out.append((lo, min(hi, r1)))
    if max(lo, r2) <= hi:
        out.append((max(lo, r2), hi))
    return out


def feasible_interval(C, c, r):
    """Exact feasible input interval of the SAFE-Switch filter for single-input plants."""
    a = C.Ph @ (C.A @ c)
    b = C.Ph @ C.B[:, 0]
    aa, ab, bb = a @ a, a @ b, b @ b
    kap = float(C.p @ C.dB[:, 0])
    K1 = C.ell - float(C.p @ (C.G @ r + C.dA @ np.abs(c) + C.Ew))
    K2 = C.lam * C.normP(c) + C.eps0
    um = C.umax
    if K1 < 0 or K2 < 0:
        return None
    iv = []
    # u >= 0 : sqrt(q) <= K1 - kap u
    hi_p = um if kap <= 0 else min(um, K1 / kap)
    iv += _quad_le0(bb - kap ** 2, 2 * ab + 2 * K1 * kap, aa - K1 ** 2, 0.0, hi_p)
    # u <= 0 : substitute v = -u
    iv += [(-h, -l) for (l, h) in _quad_le0(bb - kap ** 2, -2 * ab + 2 * K1 * kap, aa - K1 ** 2, 0.0, hi_p)]
    if not iv:
        return None
    lo1, hi1 = min(l for l, _ in iv), max(h for _, h in iv)
    iv2 = _quad_le0(bb, 2 * ab, aa - K2 ** 2, -um, um)
    if not iv2:
        return None
    lo, hi = max(lo1, iv2[0][0]), min(hi1, iv2[0][1])
    return (lo, hi) if lo <= hi else None


def _filter_1d_exact(self, c, r, uc):
    iv = feasible_interval(self, c, r)
    if iv is None:
        uf = float(np.clip(self.K @ c, -self.umax, self.umax)[0])
        return np.array([uf]), False, True
    us = min(max(uc, iv[0]), iv[1])
    return np.array([us]), True, abs(us - uc) > 1e-9


Certificate._filter_1d_bisect = Certificate._filter_1d
Certificate._filter_1d = _filter_1d_exact
