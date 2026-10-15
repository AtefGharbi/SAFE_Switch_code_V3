"""Closed-loop networked simulation with sensor/actuator channels and metrics."""
import numpy as np
from scenarios import sample, corrupt


def true_matrices(design, sc):
    pl = design.plant
    rel = sc["cfg"].rel_unc
    p = dict(pl.nominal)
    for j, key in enumerate(pl.unc_keys):
        p[key] = pl.nominal[key] * (1 + rel * sc["param_scale"][j])
    return pl.matrices(p)


def simulate(design, ctrl, sc, reward_fn=None, log=False):
    pl = design.plant
    n, m = pl.n, pl.m
    T = sc["T"]
    A, B, E = true_matrices(design, sc)
    x = sc["x0"].copy()
    arrivals = [[] for _ in range(T + 1)]
    y0, _ = corrupt(sc, 0, x, sc["v"][0]) if False else (x + sc["v"][0], None)
    ctrl.reset(sc, y0)
    sc["_ctrl"] = ctrl
    u_app = np.zeros(m)
    buf = []
    X = np.zeros((T + 1, n)); U = np.zeros(T); UC = np.zeros(T)
    MODE = np.zeros(T, int); SW = np.zeros(T, int); RISK = np.zeros(T); INTERV = np.zeros(T, bool)
    FEAS = np.ones(T, bool); TT = np.zeros(T); CONT = np.ones(T, bool); F = np.full(T, np.nan)
    RW = np.zeros(T); R_ = np.zeros((T, n)); C_ = np.zeros((T, n))
    ALARM = np.zeros(T, bool); TSAFE = np.full(T, np.nan); BUF = np.zeros(T, bool)
    X[0] = x
    for k in range(T):
        # sensor side: corrupted measurement generated at k, per-channel packet
        sc["_u_prev"] = u_app
        y, a = corrupt(sc, k, x, sc["v"][k])
        for i in range(n):
            if not sc["s_lost"][k, i]:
                ka = k + int(sc["s_delay"][k, i])
                if ka <= T:
                    arrivals[ka].append((i, k, float(y[i])))
        u_cmd, newbuf, info = ctrl.step(k, arrivals[k], u_app)
        # actuator side
        if not sc["a_lost"][k]:
            u_app = np.clip(np.atleast_1d(u_cmd), -design.umax, design.umax)
            buf = list(newbuf) if newbuf is not None else []
        else:
            if buf:
                u_app = np.clip(np.atleast_1d(buf.pop(0)), -design.umax, design.umax)
            # else: zero-order hold of previous applied input
        BUF[k] = bool(sc["a_lost"][k])
        w = sc["w"][k]
        if "c" in info:
            CONT[k] = bool(np.all(np.abs(x - info["c"]) <= info["r"] + 1e-9))
            R_[k], C_[k] = info["r"], info["c"]
        if "f" in info:
            F[k] = info["f"]
        U[k] = u_app[0]
        UC[k] = info.get("uc", u_app[0])
        MODE[k] = info.get("mode", 0); SW[k] = info.get("switch", 0); RISK[k] = info.get("risk", 0.0)
        INTERV[k] = info.get("interv", False); FEAS[k] = info.get("feas", True); TT[k] = info["t"]
        ALARM[k] = info.get("alarm", False); TSAFE[k] = info.get("tsafe", np.nan)
        if reward_fn is not None:
            RW[k] = reward_fn(x, u_app, info, design)
        x = A @ x + B @ u_app + E @ w
        X[k + 1] = x
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
            X[k + 1:] = np.nan
            break
    out = dict(X=X, U=U, UC=UC, MODE=MODE, SW=SW, RISK=RISK, INTERV=INTERV, FEAS=FEAS, T=TT,
               CONT=CONT, F=F, RW=RW, R=R_, C=C_, attack=sc["attack_on"], ALARM=ALARM, TSAFE=TSAFE, BUF=BUF)
    return out


def metrics(design, out, Ts=None, band=0.1):
    Ts = design.Ts if Ts is None else Ts
    X = out["X"][:-1]
    T = len(X)
    finite = np.all(np.isfinite(X), 1)
    diverged = not finite.all()
    Xf = np.where(np.isfinite(X), X, 1e3)
    err = np.linalg.norm(Xf, axis=1)
    viol = np.any(np.abs(Xf) > design.xmax, 1)
    U = out["U"]
    cost = float(np.sum(np.sum(Xf ** 2, 1) + 0.01 * U ** 2) * Ts)
    # recovery time: after each attack interval ends, time until ||x|| < band and stays 10 steps
    att = out["attack"]
    rec = []
    ends = [k for k in range(1, T) if att[k - 1] and not att[k]]
    for e in ends:
        kk = e
        while kk < T - 10 and not np.all(err[kk:kk + 10] < max(band, 0)):
            kk += 1
        if kk < T - 10:
            rec.append((kk - e) * Ts)
    return dict(
        rmse=float(np.sqrt(np.mean(err ** 2))),
        rmse_x2=float(np.sqrt(np.mean(Xf[:, 1] ** 2))),
        max_dev=float(np.max(err)),
        viol_frac=float(np.mean(viol)),
        viol_any=float(np.any(viol)),
        diverged=float(diverged),
        energy=float(np.sum(U ** 2) * Ts),
        cost=cost,
        switches=int(np.sum(out["SW"])),
        interv=float(np.mean(out["INTERV"])),
        infeas=float(np.mean(~out["FEAS"])),
        contain=float(np.mean(out["CONT"])),
        recovery=float(np.mean(rec)) if rec else np.nan,
        t_ms=float(np.mean(out["T"]) * 1e3),
        t_max_ms=float(np.max(out["T"]) * 1e3),
        frac_R=float(np.mean(out["MODE"] == 1)),
        frac_recovery=float(np.mean(out["MODE"] >= 2)), frac_hold=float(np.mean(out["MODE"] == 3)),
        alarm_frac=float(np.mean(out.get("ALARM", np.zeros(1)))),
        **budget_check(design, out, viol, Ts),
    )


def budget_check(design, out, viol, Ts):
    """For every alarm onset: certified budget T_safe and steps until the next violation."""
    if "TSAFE" not in out:
        return {}
    on = np.where(np.isfinite(out["TSAFE"]))[0]
    T = len(viol)
    sound, budgets = True, []
    for k0 in on:
        tb = int(out["TSAFE"][k0]); budgets.append(tb)
        v = np.where(viol[k0:])[0]
        if len(v) and v[0] < tb:
            # only meaningful if containment held at onset and the fallback was applied throughout
            if out["CONT"][k0] and np.all(out["MODE"][k0:k0 + v[0] + 1] >= 2):
                sound = False
    return dict(n_alarm=int(len(on)), budget_mean_s=float(np.mean(budgets) * Ts) if budgets else np.nan,
                budget_min_s=float(np.min(budgets) * Ts) if budgets else np.nan, budget_sound=float(sound))
