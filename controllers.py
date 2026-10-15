"""Controllers: SAFE-Switch (and its ablations) and baselines.

Every controller implements
    reset(design, sc, y0)
    step(k, packets, u_prev_applied) -> (u_cmd, buffer, info)
where packets = list of (channel, generation time tau, value) arriving at time k,
u_prev_applied is the input actually applied at k-1 (TCP-like acknowledgement), and
buffer is an optional list of N_a fallback inputs for the actuator (packetised
transmission). info is a dict that is logged.
"""
import time
import numpy as np
from estimators import IntervalEstimator, KalmanOOSM
from config import dgame_are, cgame_are

W_VAR = 0.45 ** 2  # nominal disturbance variance used by the Kalman filter


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class RiskModel:
    """Calibrated logistic attack-risk model r_k = sigmoid(theta^T [1, phi_k])."""
    FEATURES = ["zmax_ewma", "rej_ewma", "kf_gap", "kf_out", "info_age", "loss_ewma", "radius", "cusum"]

    def __init__(self, theta=None, mu=None, sd=None):
        self.theta, self.mu, self.sd = theta, mu, sd

    def __call__(self, phi):
        if self.theta is None:
            return 0.0
        z = (phi - self.mu) / self.sd
        return float(_sigmoid(self.theta[0] + z @ self.theta[1:]))


class SafeSwitch:
    """Full SAFE-Switch and its ablations.

    supervisor : 'ppo' | 'threshold' | 'always_N' | 'always_R'
    """

    def __init__(self, design, supervisor="ppo", policy=None, risk=None, use_filter=True,
                 timestamp_aware=True, dwell=10, packetized=True, risk_in_obs=True,
                 thr_on=0.5, thr_off=0.3, unc_on=0.6, name="SAFE-Switch", deterministic=True,
                 use_zono=True, quarantine=True, reanchor=True, gross_emax=None, auth_period=None,
                 monitor=True, recovery=False):
        self.recovery = recovery
        self.est_kw = dict(use_zono=use_zono, quarantine=quarantine, gross_emax=gross_emax,
                           auth_period=auth_period)
        self.monitor = monitor
        self.reanchor = reanchor
        self.D = design
        self.sup = supervisor
        self.policy = policy
        self.risk = risk if risk is not None else RiskModel()
        self.use_filter = use_filter
        self.ts = timestamp_aware
        self.dwell = dwell
        self.packetized = packetized
        self.risk_in_obs = risk_in_obs
        self.thr_on, self.thr_off, self.unc_on = thr_on, thr_off, unc_on
        self.name = name
        self.deterministic = deterministic
        self.trace = []  # for PPO: (obs, action, logp, value)
        self.rng = np.random.default_rng(0)

    def reset(self, sc, y0):
        D = self.D
        C = D.cert
        n = D.plant.n
        H = D.L + 4
        r0 = np.full(n, D.vbar + D.abar)
        self.est = IntervalEstimator(C, np.full(n, D.vbar + D.abar), H, y0, r0, self.ts, **self.est_kw)
        if not self.reanchor:
            self.est.N_reset = 0
        self.kf = KalmanOOSM(D.plant.A, D.plant.B, D.plant.E, W_VAR, D.vbar ** 2 / 3, y0,
                             np.eye(n) * (D.vbar ** 2), H, self.ts)
        self.mode = 1 if self.sup == "always_R" else 0
        self.t_in_mode = self.dwell
        self.zmax_e = 0.0
        self.rej_e = 0.0
        self.loss_e = 0.0
        self.cusum = 0.0
        self.trace = []
        self.n = n
        self._alarm_prev = False
        self._in_recovery = False
        self._rec_law = "KR"
        self._sent = []
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def features(self, k, stats, n_arr):
        D, C = self.D, self.D.cert
        c, r = self.est.box
        xk = self.kf.x
        stats = [(s.replace("mon_", ""), z) for (s, z) in stats]
        zs = [z for (s, z) in stats if s in ("acc", "rej")]
        zmax = max(zs) if zs else 0.0
        if zs:
            self.zmax_e = 0.8 * self.zmax_e + 0.2 * min(zmax, 5.0)
            self.rej_e = 0.8 * self.rej_e + 0.2 * np.mean([s == "rej" for (s, _) in stats if s != "old"])
            for z in zs:
                self.cusum = min(10.0, max(0.0, self.cusum + min(z, 5.0) - 0.5))
        else:
            self.zmax_e *= 0.95
            self.rej_e *= 0.95
        self.loss_e = 0.9 * self.loss_e + 0.1 * (1 - min(n_arr, self.n) / self.n)
        kf_gap = min(C.normP(xk - c) / D.ell, 3.0)
        kf_out = float(np.max(np.maximum(np.abs(xk - c) - r, 0) / C.rbar))
        info_age = (k - np.min(self.est.last_tau)) / D.L
        radius = float(C.p @ r) / C.qbar
        return np.array([self.zmax_e, self.rej_e, kf_gap, min(kf_out, 3.0), min(info_age, 3.0),
                         self.loss_e, min(radius, 3.0), self.cusum / 10.0])

    def observation(self, phi, risk, c, r):
        D, C = self.D, self.D.cert
        f = C.f(c, r) / D.ell
        o = [*(c / D.xmax), *(r / C.rbar), f, risk, phi[2], phi[4], phi[5], float(self.mode),
             min(self.t_in_mode / max(self.dwell, 1), 1.0), phi[1]]
        o = np.array(o)
        if not self.risk_in_obs:
            o[[5, 6, 11]] = 0.0
        return np.clip(o, -5, 5)

    def step(self, k, packets, u_prev):
        t0 = time.perf_counter()
        D, C = self.D, self.D.cert
        if k > 0:
            self.est.time_update(k, u_prev)
            self.kf.time_update(k, u_prev)
        stats = [self.est.measure(i, tau, y) for (i, tau, y) in sorted(packets, key=lambda p: p[1])]
        self.kf.measure_batch(packets)
        c, r = self.est.box
        phi = self.features(k, stats, len(packets))
        risk = self.risk(phi)
        obs = self.observation(phi, risk, c, r)
        # ---- supervisor
        prev = self.mode
        if self.sup == "always_N":
            a = 0
        elif self.sup == "always_R":
            a = 1
        elif self.sup == "random":
            a = 1 - prev if self.rng.random() < 0.05 else prev
        elif self.sup == "schedule":  # externally prescribed (e.g. adversarial) switching signal
            a = int(self.mode_schedule[min(k, len(self.mode_schedule) - 1)])
        elif self.sup == "threshold":
            unc = C.f(c, r) / D.ell
            if self.mode == 0:
                a = 1 if (risk > self.thr_on or unc > self.unc_on) else 0
            else:
                a = 0 if (risk < self.thr_off and unc < self.unc_on) else 1
        else:
            a, logp, v = self.policy.act(obs, deterministic=self.deterministic)
            self.trace.append((obs, a, logp, v))
        # dwell-time: leaving the robust mode requires tau_min steps; entering it is immediate
        if prev == 1 and a == 0 and self.t_in_mode < self.dwell:
            a = 1
        self.mode = a
        self.t_in_mode = self.t_in_mode + 1 if a == prev else 0
        # ---- candidate action
        if a == 0:
            uc = self.D.KN @ self.kf.x
        else:
            uc = self.D.KM @ c
        uc = np.atleast_1d(uc)
        # ---- runtime assumption monitor (Corollary 1). V3: the age threshold is N_b + d_max, the age
        # that A4 guarantees; the radius and level checks use the certified set I.
        alarm_pre = False
        if self.monitor:
            gap = k - np.min(self.est.last_tau)
            alarm_pre = bool(gap > D.Nb + D.dmax or np.any(r > C.rbar + 1e-9) or C.f(c, r) > D.ell + 1e-9)
        # ---- recovery mode (Proposition 2). V3: starts at the alarm step itself, applies sat(K_R c)
        # WITHOUT the filter, and stays active until the monitor is silent and (c, r) is back in I.
        in_rec = self.recovery and (alarm_pre or getattr(self, "_in_recovery", False))
        tsafe_rec = np.nan
        if in_rec and not getattr(self, "_in_recovery", False):
            # recovery onset: pick the recovery law with the larger certified budget (Proposition 2)
            self._rec_law, tsafe_rec = self.choose_recovery_law(k, c, r, extra=self._pending_buffers(k))
        if in_rec:
            if getattr(self, "_rec_law", "KR") == "hold":
                uc = np.zeros(D.plant.m); a = 3          # mode 3: recovery by open-loop hold
            else:
                uc = np.atleast_1d(np.clip(C.K @ c, -C.umax, C.umax)); a = 2   # mode 2: fallback law
            us, feas, interv = uc, True, False
        elif self.use_filter:
            us, feas, interv = C.filter(c, r, uc)
            if self.recovery and not feas:          # infeasible filter (M4) = alarm: recovery starts now
                in_rec, a = True, 2
                self._rec_law, tsafe_rec = self.choose_recovery_law(k, c, r, extra=self._pending_buffers(k))
                if self._rec_law == "hold":
                    us = np.zeros(D.plant.m); a = 3
        else:
            us, feas, interv = np.clip(uc, -D.umax, D.umax), True, False
        # ---- packetised fallback sequence for actuator-side losses (saturated at the certified ubar)
        buf = None
        if self.packetized:
            buf = []
            cc = C.A @ c + C.B @ us
            hold = in_rec and getattr(self, "_rec_law", "KR") == "hold"
            for _ in range(D.Na):
                uf = np.zeros(D.plant.m) if hold else np.clip(D.KR @ cc, -C.umax, C.umax)
                buf.append(uf)
                cc = C.A @ cc + C.B @ uf
        self._sent = (getattr(self, "_sent", []) + [(k, buf)])[-D.Na:]
        # ---- alarm bookkeeping and certified time budget at alarm onset
        alarm, tsafe = False, np.nan
        if self.monitor:
            alarm = bool(alarm_pre or not feas)
            if alarm and not getattr(self, "_alarm_prev", False):
                tsafe = tsafe_rec if np.isfinite(tsafe_rec) else \
                    self.time_budget(c, r, u0=us, extra=self._pending_buffers(k))
            self._alarm_prev = alarm
        if self.recovery:
            # leave recovery only when the monitor is silent, i.e. age <= N_b + d_max and (c, r) in I
            self._in_recovery = alarm
            self.est.N_reset = 0 if self._in_recovery else 20
        dt = time.perf_counter() - t0
        info = dict(alarm=alarm, tsafe=tsafe, mode=a, switch=int(a != prev), risk=risk, uc=float(uc[0]), us=float(us[0]),
                    feas=feas, interv=interv, c=c.copy(), r=r.copy(), f=C.f(c, r),
                    n_rej=sum(s in ("rej", "mon_rej") for s, _ in stats),
                    n_acc=sum(s in ("acc", "mon_acc") for s, _ in stats),
                    phi=phi, obs=obs, t=dt)
        return us, buf, info


def _pending_buffers(self, k):
    """Inputs that the actuator may still apply at steps k, k+1, ... from packets sent before k (A6):
    element j holds the hull (lo, hi) of buffer entries buf_{k'}[k + j - k' - 1], or None."""
    D = self.D
    ex = []
    for j in range(D.Na):
        vals = [b[k + j - kp - 1] for (kp, b) in getattr(self, "_sent", [])
                if b and kp < k and 0 <= k + j - kp - 1 < len(b)]
        ex.append((np.min(vals, axis=0), np.max(vals, axis=0)) if vals else None)
    return ex


SafeSwitch._pending_buffers = _pending_buffers


def _time_budget(self, c, r, horizon=400, u0=None, extra=None):
    """Certified time budget (Proposition 2): number of steps for which every state reachable
    from B(c, r) stays in X_safe when the input u0 (already applied) is followed by the recovery
    law sat(K_R c'), with c' ANY centre inside the current tube box (whatever is measured later).
    The input uncertainty |K_R| r is propagated through |B| + delta_B (sat is non-expansive)."""
    from post_alarm import tube
    T = tube(self.D.cert, self.D.cert.umax, self.D.xmax, c, r, "KR", horizon, u0=u0, extra=extra)
    return horizon if np.isinf(T) else int(T)


SafeSwitch.time_budget = _time_budget


def _choose_recovery_law(self, k, c, r, horizon=400, extra=None):
    """Proposition 2 (V3). Budget of the K_R law: the tube budget, or the grace window
    W = L + N_g - age + 2 when the alarm is an age alarm raised from inside I (the information state
    then stays in I_g, Section VII); budget of the hold u = 0: its tube budget. Returns the better law."""
    from post_alarm import tube
    D, C = self.D, self.D.cert
    uK = np.clip(C.K @ c, -C.umax, C.umax)
    T_kr = tube(C, C.umax, D.xmax, c, r, "KR", horizon, u0=uK, extra=extra)
    T_hold = tube(C, C.umax, D.xmax, c, r, "hold", horizon, u0=np.zeros(D.plant.m), extra=extra)
    T_kr, T_hold = min(T_kr, horizon), min(T_hold, horizon)
    # the choice uses only the unconditional tube budgets of Proposition 2(a); the grace window of
    # Proposition 2(b) is conditional on A5 after the alarm and is reported separately (post_alarm.py)
    return ("hold", int(T_hold)) if T_hold > T_kr else ("KR", int(T_kr))


SafeSwitch.choose_recovery_law = _choose_recovery_law


class LQRKF:
    """Nominal LQR on a timestamp-aware Kalman filter (no gating, no filter)."""
    name = "LQR"

    def __init__(self, design):
        self.D = design

    def reset(self, sc, y0):
        D = self.D
        n = D.plant.n
        self.kf = KalmanOOSM(D.plant.A, D.plant.B, D.plant.E, W_VAR, D.vbar ** 2 / 3, y0,
                             np.eye(n) * D.vbar ** 2, D.L + 4)

    def step(self, k, packets, u_prev):
        t0 = time.perf_counter()
        if k > 0:
            self.kf.time_update(k, u_prev)
        self.kf.measure_batch(packets)
        u = np.clip(self.D.KN @ self.kf.x, -self.D.umax, self.D.umax)
        return u, None, dict(mode=0, switch=0, interv=False, feas=True, t=time.perf_counter() - t0,
                             risk=0.0)


class HinfSet:
    """Discrete H-infinity (game) state feedback on the gated interval estimator centre."""
    name = "Robust H∞"

    def __init__(self, design, gamma=2.0):
        self.D = design
        pl = design.plant
        self.K, _, _ = dgame_are(pl.A, pl.B, pl.E, np.eye(pl.n), np.eye(pl.m), gamma)

    def reset(self, sc, y0):
        D = self.D
        n = D.plant.n
        self.est = IntervalEstimator(D.cert, np.full(n, D.vbar + D.abar), D.L + 4, y0,
                                     np.full(n, D.vbar + D.abar))

    def step(self, k, packets, u_prev):
        t0 = time.perf_counter()
        if k > 0:
            self.est.time_update(k, u_prev)
        for (i, tau, y) in sorted(packets, key=lambda p: p[1]):
            self.est.measure(i, tau, y)
        c, r = self.est.box
        u = np.clip(self.K @ c, -self.D.umax, self.D.umax)
        return u, None, dict(mode=1, switch=0, interv=False, feas=True, t=time.perf_counter() - t0,
                             risk=0.0, f=self.D.cert.f(c, r))


class HuangDong2020:
    """Switched H-infinity controller with prescribed-performance switching (Huang & Dong 2020).

    Faithful elements: mode-0 game-ARE pair (A, Bu, Bd), mode-1 pair for the compromised
    model (Lam A Lam^-1, Lam Bu, Lam Bd), feedback on the received (possibly compromised)
    state x~, switching law (33) with eps_i = eta0 exp(-a_i (t - t_k')) V_i(t_k') + eta_inf.
    Adaptation for networks: hold-last-received measurement per channel (no timestamps).
    The ADP identifier/critic is replaced by its converged (model-based) solution, which is
    favourable to the baseline (Remarks 6-7 of [1]).
    """
    name = "Huang–Dong [1]"

    def __init__(self, design, Lam=(-1.5, -0.5), gamma=2.0, eta0=1.1, eta_inf=0.05, a=(2.6, 1.7),
                 continuous_gains=True):
        self.D = design
        pl = design.plant
        Lm = np.diag(Lam)
        self.Lam = Lm
        if continuous_gains:
            K0, _, P0 = cgame_are(pl.Ac, pl.Bc, pl.Dc, np.eye(pl.n), 1.0, gamma)
            At = Lm @ pl.Ac @ np.linalg.inv(Lm)
            K1, _, P1 = cgame_are(At, Lm @ pl.Bc, Lm @ pl.Dc, np.eye(pl.n), 1.0, gamma)
            self.K = [-K0, -K1]
        else:
            K0, _, P0 = dgame_are(pl.A, pl.B, pl.E, np.eye(pl.n), np.eye(pl.m), gamma)
            At = Lm @ pl.A @ np.linalg.inv(Lm)
            K1, _, P1 = dgame_are(At, Lm @ pl.B, Lm @ pl.E, np.eye(pl.n), np.eye(pl.m), gamma)
            self.K = [K0, K1]
        self.P = [P0, P1]
        self.eta0, self.eta_inf, self.a = eta0, eta_inf, a
        self.Ts = pl.Ts

    def reset(self, sc, y0):
        self.xt = np.array(y0, float)
        self.mode = 0
        self.tk = 0
        self.Vk = float(self.xt @ self.P[0] @ self.xt)

    def step(self, k, packets, u_prev):
        t0 = time.perf_counter()
        for (i, tau, y) in packets:   # latest arrival overwrites (no timestamp handling)
            self.xt[i] = y
        i = self.mode
        V = float(self.xt @ self.P[i] @ self.xt)
        eps = self.eta0 * np.exp(-self.a[i] * (k - self.tk) * self.Ts) * self.Vk + self.eta_inf
        sw = 0
        if V > eps:
            self.mode = 1 - i
            self.tk = k
            self.Vk = float(self.xt @ self.P[self.mode] @ self.xt)
            sw = 1
        u = np.clip(self.K[self.mode] @ self.xt, -self.D.umax, self.D.umax)
        return u, None, dict(mode=self.mode, switch=sw, interv=False, feas=True,
                             t=time.perf_counter() - t0, risk=0.0)


class RobustMPC:
    """Tube-type robust MPC on the interval-estimator centre with tightened constraints
    and the certified terminal set (SOC terminal constraint)."""
    name = "Robust MPC"

    def __init__(self, design, N=10, Q=None, R=1.0):
        import cvxpy as cp
        self.D = design
        pl, C = design.plant, design.cert
        n, m = pl.n, pl.m
        self.N = N
        Q = np.diag([1.0, 100.0]) if Q is None else Q
        self.c0 = cp.Parameter(n)
        self.xlim = cp.Parameter((N + 1, n), nonneg=True)
        self.term = cp.Parameter(nonneg=True)
        self.u = cp.Variable((N, m))
        X = cp.Variable((N + 1, n))
        cost = 0
        cons = [X[0] == self.c0]
        for j in range(N):
            cons += [cp.abs(self.u[j]) <= design.umax]
            cons += [X[j + 1] == C.A @ X[j] + C.B @ self.u[j]]
            cost += cp.quad_form(X[j + 1], Q) + R * cp.sum_squares(self.u[j])
            cons += [cp.abs(X[j + 1]) <= self.xlim[j + 1]]
        cons += [cp.norm(C.Ph @ X[N]) <= self.term]
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def reset(self, sc, y0):
        D = self.D
        n = D.plant.n
        self.est = IntervalEstimator(D.cert, np.full(n, D.vbar + D.abar), D.L + 4, y0,
                                     np.full(n, D.vbar + D.abar))
        self.n_inf = 0

    def step(self, k, packets, u_prev):
        t0 = time.perf_counter()
        D, C = self.D, self.D.cert
        if k > 0:
            self.est.time_update(k, u_prev)
        for (i, tau, y) in sorted(packets, key=lambda p: p[1]):
            self.est.measure(i, tau, y)
        c, r = self.est.box
        # open-loop radius growth (worst-case input) -> constraint tightening
        rr = r.copy()
        xl = np.zeros((self.N + 1, len(c)))
        cc = np.abs(c)
        for j in range(self.N + 1):
            xl[j] = np.maximum(D.xmax - rr, 1e-3)
            rr = C.G @ rr + C.dA @ (C.sig * D.ell) + C.dB @ np.full(D.plant.m, D.umax) \
                + C.Ew
        self.c0.value = c
        self.xlim.value = xl
        self.term.value = max(D.ell - float(C.p @ r), 1e-3)
        feas = True
        try:
            self.prob.solve(solver="CLARABEL", warm_start=True)
            ok = self.prob.status in ("optimal", "optimal_inaccurate")
        except Exception:
            ok = False
        if ok:
            u = np.asarray(self.u.value[0]).ravel()
        else:
            feas = False
            u = np.clip(D.KR @ c, -D.umax, D.umax)
        return np.clip(u, -D.umax, D.umax), None, dict(mode=1, switch=0, interv=False, feas=feas,
                                                         t=time.perf_counter() - t0, risk=0.0)
