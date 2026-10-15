"""Post-alarm guarantees (Proposition 2, V3).

(a) Runtime budget. Tube of boxes from the information state at the alarm: the first step uses the
    input actually applied at the alarm step; afterwards the recovery law is applied for ANY centre c'
    inside the current tube box (whatever is measured later):
      * law "KR":   u = sat(K_R c')  -> input centre sat(K_R c_hat), input radius |K_R| r_hat
                    (sat is non-expansive, so |sat(K_R c') - sat(K_R c_hat)| <= |K_R| |c' - c_hat|);
      * law "hold": u = 0            -> no input uncertainty (open-loop hold).
    The budget is the number of steps for which every box of the tube lies in X_safe.
(b) Uniform grace window (design time). The monitor alarms as soon as the age of the last accepted
    packet exceeds N_b + d_max. If (C1)-(C5) also hold for the radius bound rbar_g computed with the
    horizon L + N_g, then from ANY alarm (under A1-A3, A5, A6) the information state stays in
    I_g = {f <= ell, r <= rbar_g} for N_a + N_g steps under the recovery law sat(K_R c) = K_R c, and the
    state stays in Omega for W = N_a + N_g + 1 steps, irrespective of later measurements.
(c) Hold certificate (design time): if rho(G) < 1, the hold tube converges; we report its limit radius
    and the fraction of sampled information states of I from which the hold is safe for all time.
"""
import json, os
import numpy as np
from safety import radius_bound

OUT = os.path.join(os.path.dirname(__file__), "results")


def tube(C, ubar, xmax, c, r, law="KR", horizon=2000, u0=None, binv=None, extra=None):
    """Number of steps j for which all boxes B_0..B_{j-1} lie in X_safe (B_0 = B(c, r)); inf if never left
    (for the hold: once |c| + r lies in the hold-invariant box binv, the tube never leaves X_safe).
    extra[j] = (lo, hi): additional inputs that the actuator may still apply at step j (buffered inputs of
    packets sent before the alarm, A6); the input set at step j is the hull of both."""
    absK, absB, Ew = np.abs(C.K), np.abs(C.B), C.Ew
    if law == "hold" and binv is None:
        binv = getattr(C, "_binv", "unset")
        if isinstance(binv, str):
            binv = C._binv = hold_invariant_box(C, xmax)
    c, r = np.array(c, float), np.array(r, float)
    m = C.B.shape[1]
    for j in range(horizon):
        if np.any(np.abs(c) + r > xmax):
            return j
        if law == "hold" and binv is not None and (j > 0 or u0 is None) and \
                (extra is None or j >= len(extra)) and np.all(np.abs(c) + r <= binv):
            return np.inf
        if j == 0 and u0 is not None:                       # input already applied at the alarm step
            u = np.atleast_1d(np.asarray(u0, float)); ur = np.zeros(m)
        elif law == "KR":
            u = np.clip(C.K @ c, -ubar, ubar); ur = absK @ r
        else:
            u = np.zeros(m); ur = np.zeros(m)
        if extra is not None and j < len(extra) and extra[j] is not None:
            lo = np.minimum(u - ur, extra[j][0]); hi = np.maximum(u + ur, extra[j][1])
            u, ur = 0.5 * (lo + hi), 0.5 * (hi - lo)
        c, r = C.A @ c + C.B @ u, C.G @ r + C.dA @ np.abs(c) + C.dB @ (np.abs(u) + ur) + absB @ ur + Ew
        if not np.all(np.isfinite(r)) or np.max(r) > 1e6:
            return j + 1
    return np.inf


def grace_window(D, Ng_max=60):
    """Largest N_g for which the certificate (C2)-(C3) still holds with horizon L + N_g."""
    C, ell = D.cert, D.ell
    best, rows = None, []
    for Ng in range(0, Ng_max + 1):
        rb, ok = radius_bound(C.G, D.plant.dA, D.plant.dB, D.plant.E, C.wbar, C.sig * ell, D.ufilt,
                              np.full(D.plant.n, (D.vbar + D.abar) * (1 + 1e-9) + 1e-8 * float(np.max(D.xmax))), D.L + Ng, iters=300,
                              dE=D.plant.dE)
        q = float(C.p @ rb)
        lhs = C.alpha * ell + max(C.beta - C.alpha, 0.0) * q + C.c0
        good = bool(ok and lhs <= ell and q < ell)
        rows.append(dict(Ng=Ng, rbar=rb.tolist(), qbar=q, C2_margin=ell - lhs, holds=good))
        if not good:
            break
        best = Ng
    W = None if best is None else D.Na + best + 1
    return dict(Ng=best, W_steps=W, W_s=None if W is None else W * D.Ts, table=rows)


def hold_analysis(D, n=1000, seed=0, horizon=1000):
    C = D.cert
    rhoG = float(max(abs(np.linalg.eigvals(C.G))))
    out = dict(rho_G=rhoG)
    if rhoG < 1:
        rinf = np.linalg.solve(np.eye(len(C.G)) - C.G, C.Ew)
        out["r_inf"] = rinf.tolist()
        out["r_inf_inside"] = bool(np.all(rinf < D.xmax))
    rng = np.random.default_rng(seed)
    Phi = np.linalg.inv(C.Ph)
    inf_hold = inf_kr = 0
    Th, Tk = [], []
    for _ in range(n):
        r = rng.uniform(0, 1, len(C.G)) * C.rbar
        s = rng.uniform(0, 1) * max(D.ell - C.p @ r, 0)
        d = rng.normal(size=len(C.G)); d /= np.linalg.norm(d)
        c = Phi @ (d * s)
        ke = np.full(C.B.shape[1], C.kap * D.ell)
        ex = [(-ke, ke)] * D.Na                                   # pre-alarm buffered inputs (Lemma 3)
        th = tube(C, D.ufilt, D.xmax, c, r, "hold", horizon, u0=np.zeros(C.B.shape[1]), extra=ex)
        tk = tube(C, D.ufilt, D.xmax, c, r, "KR", horizon, u0=np.clip(C.K @ c, -D.ufilt, D.ufilt), extra=ex)
        inf_hold += np.isinf(th); inf_kr += np.isinf(tk)
        Th.append(th); Tk.append(tk)
    fin = lambda T: [t for t in T if np.isfinite(t)]
    out.update(n_sampled=n, frac_hold_forever=inf_hold / n, frac_KR_forever=inf_kr / n,
               median_T_hold_s=float(np.median(Th) * D.Ts) if np.isfinite(np.median(Th)) else None,
               median_T_KR_s=float(np.median(Tk) * D.Ts) if np.isfinite(np.median(Tk)) else None,
               min_T_hold_s=float(min(Th) * D.Ts), min_T_KR_s=float(min(Tk) * D.Ts),
               frac_max_law_forever=float(np.mean([np.isinf(a) or np.isinf(b) for a, b in zip(Th, Tk)])))
    return out


def hold_invariant_box(C, xmax):
    """Largest box {|x| <= b} along b_inf + t v (v: Perron vector of G) with G b + Ew <= b and b <= xmax.
    Under the hold u = 0 this box is robustly invariant: |x+| <= G |x| + Ew componentwise."""
    w, V = np.linalg.eig(C.G)
    i = int(np.argmax(np.abs(w)))
    if abs(w[i]) >= 1:
        return None
    v = np.abs(np.real(V[:, i])); v /= v.max()
    binf = np.linalg.solve(np.eye(len(C.G)) - C.G, C.Ew)
    if np.any(binf > xmax):
        return None
    t = float(np.min((xmax - binf) / np.maximum(v, 1e-12)))
    b = binf + t * v
    if not (np.all(C.G @ b + C.Ew <= b + 1e-12) and np.all(b <= xmax + 1e-12)):
        return None                                   # invariance not verified: no claim
    return b


def uniform_hold(D, ngrid=121, horizon=4000):
    """Rigorous lower bound on the hold budget over all of I (cover of the centre set by boxes,
    inclusion-monotonicity of the tube). Step 0 needs no check (the state is in Omega by Theorem 1).
    A cell is safe forever once its tube bound |c|+r enters the invariant box."""
    C, ell, xmax = D.cert, D.ell, D.xmax
    binv = hold_invariant_box(C, xmax)
    ext = C.sig * ell
    xs = np.linspace(-ext[0], ext[0], ngrid); ys = np.linspace(-ext[1], ext[1], ngrid)
    h = np.array([xs[1] - xs[0], ys[1] - ys[0]]) / 2
    worst, n_cells = np.inf, 0
    for a in xs:
        for b in ys:
            ci = np.array([a, b])
            smin = C.normP(ci) - float(C.p @ h)
            if smin > ell:
                continue
            n_cells += 1
            r = np.minimum(C.rbar, np.maximum(ell - max(smin, 0.0), 0.0) / C.p) + h
            c = ci.copy(); T = np.inf
            ke = np.full(C.B.shape[1], C.kap * ell)
            for j in range(horizon):
                if j > 0 and np.any(np.abs(c) + r > xmax):
                    T = j; break
                if j >= D.Na and binv is not None and np.all(np.abs(c) + r <= binv):
                    break                                   # forever
                if j < D.Na:     # input in hull{0, pre-alarm buffered inputs |u| <= kappa*ell}
                    c, r = C.A @ c, C.G @ r + C.dA @ np.abs(c) + (C.dB + np.abs(C.B)) @ ke + C.Ew
                else:
                    c, r = C.A @ c, C.G @ r + C.dA @ np.abs(c) + C.Ew
            else:
                T = horizon
            worst = min(worst, T)
    return dict(T_hold_min_steps=None if np.isinf(worst) else int(worst),
                T_hold_min_s=None if np.isinf(worst) else worst * D.Ts, forever=bool(np.isinf(worst)),
                invariant_box=None if binv is None else binv.tolist(), n_cells=n_cells)


if __name__ == "__main__":
    from bench_designs import make_design
    res = {}
    for b in ["dcmotor", "tank"]:
        D = make_design(b)
        g = grace_window(D)
        h = hold_analysis(D)
        u = uniform_hold(D)
        res[b] = dict(Ts=D.Ts, Na=D.Na, Nb=D.Nb, dmax=D.dmax, grace=g, hold=h, hold_uniform=u)
        print(b, {k: g[k] for k in ("Ng", "W_steps", "W_s")}, h, u, flush=True)
    json.dump(res, open(os.path.join(OUT, "post_alarm.json"), "w"), indent=1, default=float)
