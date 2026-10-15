"""State estimators.

IntervalEstimator : timestamp-aware, consistency-gated interval (box) set-membership
                    estimator with out-of-sequence re-propagation (Section IV-A).
KalmanOOSM        : timestamp-aware Kalman filter with out-of-sequence measurement
                    re-processing; no attack gating (nominal estimator).
"""
import numpy as np


class IntervalEstimator:
    """Hybrid zonotope/box set-membership estimator (Section IV-A).

    Stored per time j: zonotope Z_j = <cz_j, Gz_j> and box B_j = (c_j, r_j) with
    B_j ⊆ boxpredict(B_{j-1}) ∩ hull(Z_j) ∩ measurement strips. The zonotope keeps
    state correlations (so a speed measurement also tightens the current), the box is
    what the safety certificate uses. Packets are timestamped; out-of-sequence packets
    trigger re-propagation from their generation time (accepted measurements replayed).
    A packet is accepted only if it is consistent with the stored box at its timestamp.
    """

    def __init__(self, cert, delta_meas, H, c0, r0, timestamp_aware=True, use_zono=True,
                 max_gen=None, quarantine=True, kq=0.25, Tq=100, n_release=3, gross_emax=None,
                 auth_period=None, **_):
        self.gross_emax = gross_emax      # no re-anchoring on gross inconsistencies (persistent-attack mode)
        self.auth_period = auth_period    # fuse only authenticated packets (tau % M == 0)
        self.n_gross_skip = 0
        self.quar = quarantine
        self.kq, self.Tq, self.n_release = kq, Tq, n_release
        self.q_on = np.zeros(len(c0), bool)
        self.q_last = np.zeros(len(c0), int)
        self.q_ok = np.zeros(len(c0), int)
        self.rej_run = np.zeros(len(c0), int)
        self.N_reset = 20
        self.n_reanchor = 0
        self.C = cert
        self.delta = np.asarray(delta_meas, float).copy()
        self.H = H
        self.ts = timestamp_aware
        self.zono = use_zono
        n = len(c0)
        self.max_gen = max(8, 2 * n) if max_gen is None else max_gen
        self.n = n
        self.n_empty = 0          # V3: empty intersections (only possible if containment already failed)
        self.gate_tol = 1e-9
        self.k = 0
        self.c = {0: np.array(c0, float)}
        self.r = {0: np.array(r0, float)}
        self.Z = {0: (np.array(c0, float), np.diag(np.array(r0, float)))}
        self.meas = {0: []}
        self.u = {}
        self.last_tau = np.full(n, 0)
        self.absE_w = cert.Ew

    # ---- zonotope helpers
    def _reduce(self, G):
        if G.shape[1] <= self.max_gen:
            return G
        nrm = np.linalg.norm(G, axis=0) ** 2 / (np.max(np.abs(G), axis=0) + 1e-15)
        idx = np.argsort(nrm)
        keep = idx[-(self.max_gen - self.n):]
        boxed = idx[:-(self.max_gen - self.n)]
        return np.hstack([G[:, keep], np.diag(np.sum(np.abs(G[:, boxed]), axis=1))])

    def _zpredict(self, Z, c, r, u):
        cz, G = Z
        C = self.C
        cz2 = C.A @ cz + C.B @ u
        extra = C.dA @ (np.abs(c) + r) + C.dB @ np.abs(u) + self.absE_w
        G2 = np.hstack([C.A @ G, np.diag(extra)])
        return cz2, self._reduce(G2)

    def _zstrip(self, Z, i, y, d):
        cz, G = Z
        g = G[i, :]
        S = G @ g
        den = g @ g + d ** 2
        lam = S / den
        cz2 = cz + lam * (y - cz[i])
        G2 = np.hstack([G - np.outer(lam, g), (lam * d)[:, None]])
        return cz2, self._reduce(G2)

    @staticmethod
    def _hull(Z):
        cz, G = Z
        return cz, np.sum(np.abs(G), axis=1)

    def _intersect(self, c1, r1, c2, r2, keep=1):
        """Box intersection. V3: if it is empty (possible only when containment has already failed),
        operand `keep` is returned: the prediction for a new set (keep=1), the previously stored set
        for a recomputation (keep=2). Either way every stored set stays inside the prediction from the
        previous stored set and inside its own earlier value."""
        lo = np.maximum(c1 - r1, c2 - r2)
        hi = np.minimum(c1 + r1, c2 + r2)
        if np.any(hi < lo):
            self.n_empty += 1
            return (c1, r1) if keep == 1 else (c2, r2)
        return 0.5 * (lo + hi), 0.5 * (hi - lo)

    def _apply_meas(self, j):
        c, r = self.c[j], self.r[j]
        Z = self.Z.get(j)
        for (i, y, d) in self.meas.get(j, []):
            lo = max(c[i] - r[i], y - d)
            hi = min(c[i] + r[i], y + d)
            if hi < lo:                      # V3: inconsistent strip on replay -> ignored (flagged)
                self.n_empty += 1
                continue
            c = c.copy(); r = r.copy()
            c[i], r[i] = 0.5 * (lo + hi), 0.5 * (hi - lo)
            if self.zono:
                Z = self._zstrip(Z, i, y, d)
        if self.zono:
            hc, hr = self._hull(Z)
            c, r = self._intersect(c, r, hc, hr)
            self.Z[j] = Z
        self.c[j], self.r[j] = c, r

    def _step(self, j):
        """Recompute time j+1 from j (prediction ∩ previous knowledge at j+1)."""
        cp, rp = self.C.predict(self.c[j], self.r[j], self.u[j])
        if j + 1 in self.c:
            cp, rp = self._intersect(cp, rp, self.c[j + 1], self.r[j + 1], keep=2)
        self.c[j + 1], self.r[j + 1] = cp, rp
        if self.zono:
            self.Z[j + 1] = self._zpredict(self.Z[j], self.c[j], self.r[j], self.u[j])
        self._apply_meas(j + 1)

    def time_update(self, k, u_prev):
        self.u[k - 1] = np.atleast_1d(u_prev).astype(float)
        self.meas.setdefault(k, [])
        self.k = k
        self._step(k - 1)
        for j in list(self.c.keys()):
            if j < k - self.H:
                for dct in (self.c, self.r, self.Z, self.meas, self.u):
                    dct.pop(j, None)

    def measure(self, i, tau, y):
        if not self.ts:
            tau = self.k
        if tau < self.k - self.H or tau not in self.c:
            return "old", np.nan
        c, r = self.c[tau], self.r[tau]
        z = abs(y - c[i]) / (r[i] + self.delta[i])
        if self.auth_period and tau % self.auth_period != 0:
            return ("mon_rej" if z > 1.0 + 1e-9 else "mon_acc"), z   # unauthenticated: monitored only
        if self.quar and self.q_on[i] and self.k - self.q_last[i] > self.Tq:
            self.q_on[i] = False  # time-out release
        if z > 1.0 + self.gate_tol:   # numerical tolerance: admissible packets satisfy z <= 1 exactly
            if self.quar:
                self.q_on[i] = True
                self.q_last[i] = self.k
                self.q_ok[i] = 0
            self.rej_run[i] += 1
            if self.N_reset and self.rej_run[i] >= self.N_reset:
                excess = abs(y - c[i]) - r[i] - self.delta[i]
                if self.gross_emax is not None and excess > self.gross_emax:
                    self.n_gross_skip += 1      # gross inconsistency: keep dead-reckoning
                else:
                    self._reanchor(tau, i, y)
                self.rej_run[i] = 0
            return "rej", z
        self.rej_run[i] = 0
        if self.quar and self.q_on[i]:
            # quarantined channel: accept only packets close to the centre of the information set
            if abs(y - c[i]) > self.delta[i] + self.kq * r[i]:
                self.q_ok[i] = 0
                return "rej", max(z, 1.0 + 1e-9)
            self.q_ok[i] += 1
            if self.q_ok[i] >= self.n_release:
                self.q_on[i] = False
        self._dfuse = self.delta[i] + self.gate_tol * (r[i] + self.delta[i])
        self.meas.setdefault(tau, []).append((i, y, self._dfuse))
        if self.zono:
            applied = self._replay_point(tau, i, y)
        else:
            n0 = self.n_empty; self._apply_meas(tau); applied = self.n_empty == n0
        if not applied:                      # inconsistent with the stored set: not fused, not counted
            self.meas[tau].pop()
            return "rej", z
        if getattr(self, "defer", False):
            p_ = getattr(self, "_pending", None)
            self._pending = tau if p_ is None else min(p_, tau)   # batched re-propagation
        else:
            for j in range(tau, self.k):
                self._step(j)
        self.last_tau[i] = max(self.last_tau[i], tau)
        return "acc", z

    def _reanchor(self, tau, i, y):
        """Persistent inconsistency: replace component i at tau by the hull of the stored
        interval and the measurement interval. The hull contains x_i whenever either the
        stored set was valid or the packet is honest, so containment is restored once the
        attack has ended and is never destroyed by this operation."""
        d = self.delta[i]
        c, r = self.c[tau].copy(), self.r[tau].copy()
        lo = min(c[i] - r[i], y - d)
        hi = max(c[i] + r[i], y + d)
        c[i], r[i] = 0.5 * (lo + hi), 0.5 * (hi - lo)
        self.c[tau], self.r[tau] = c, r
        self.Z[tau] = (c.copy(), np.diag(r))
        for j in range(tau + 1, self.k + 1):
            self.c.pop(j, None); self.r.pop(j, None)
        for j in range(tau, self.k):
            self._step(j)
        self.n_reanchor += 1

    def _replay_point(self, tau, i, y):
        d = self._dfuse
        c, r = self.c[tau].copy(), self.r[tau].copy()
        lo = max(c[i] - r[i], y - d)
        hi = min(c[i] + r[i], y + d)
        if hi < lo:
            self.n_empty += 1
            return False
        c[i], r[i] = 0.5 * (lo + hi), 0.5 * (hi - lo)
        Z = self._zstrip(self.Z[tau], i, y, d)
        hc, hr = self._hull(Z)
        self.c[tau], self.r[tau] = self._intersect(c, r, hc, hr)
        self.Z[tau] = Z
        return True

    def flush(self):
        """Batched timestamp update: one re-propagation from the earliest fused packet."""
        p = getattr(self, "_pending", None)
        if p is not None:
            for j in range(p, self.k):
                self._step(j)
            self._pending = None

    def measure_batch(self, packets):
        self.defer = True
        out = [self.measure(i, tau, y) for (i, tau, y) in sorted(packets, key=lambda q: q[1])]
        self.defer = False
        self.flush()
        return out

    @property
    def box(self):
        return self.c[self.k], self.r[self.k]


class KalmanOOSM:
    def __init__(self, A, B, E, w_var, v_var, x0, P0, H, timestamp_aware=True):
        self.A, self.B, self.E = A, B, E
        self.Q = E @ E.T * w_var
        self.Rv = v_var
        self.H = H
        self.ts = timestamp_aware
        self.k = 0
        self.prior = {0: (np.array(x0, float), np.array(P0, float))}
        self.post = {}
        self.meas = {0: []}
        self.u = {}
        self._update(0)

    def _update(self, j):
        x, P = self.prior[j]
        for (i, y) in self.meas.get(j, []):
            S = P[i, i] + self.Rv
            Kg = P[:, i] / S
            x = x + Kg * (y - x[i])
            P = P - np.outer(Kg, P[i, :])
        self.post[j] = (x, P)

    def time_update(self, k, u_prev):
        self.u[k - 1] = np.atleast_1d(u_prev).astype(float)
        x, P = self.post[k - 1]
        self.prior[k] = (self.A @ x + self.B @ self.u[k - 1], self.A @ P @ self.A.T + self.Q)
        self.meas.setdefault(k, [])
        self.k = k
        self._update(k)
        for j in list(self.prior.keys()):
            if j < k - self.H:
                self.prior.pop(j); self.post.pop(j, None); self.meas.pop(j, None); self.u.pop(j, None)

    def measure_batch(self, packets):
        """packets: list of (i, tau, y)."""
        if not packets:
            return
        taus = []
        for (i, tau, y) in packets:
            if not self.ts:
                tau = self.k
            if tau < self.k - self.H or tau not in self.prior:
                continue
            self.meas.setdefault(tau, []).append((i, y))
            taus.append(tau)
        if not taus:
            return
        t0 = min(taus)
        self._update(t0)
        for j in range(t0, self.k):
            x, P = self.post[j]
            self.prior[j + 1] = (self.A @ x + self.B @ self.u[j], self.A @ P @ self.A.T + self.Q)
            self._update(j + 1)

    @property
    def x(self):
        return self.post[self.k][0]
