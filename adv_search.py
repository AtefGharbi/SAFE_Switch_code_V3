"""E11: certificate validation by adversarial search (cross-entropy method).

The adversary controls, within the admissible sets of A1-A6, every exogenous quantity:
initial state on the boundary of the certified set, time-varying model error
Delta A_k, Delta B_k (element-wise box), disturbance w_k, noise v_k, corruption a_k of
every packet (|a| <= abar, i.e. A5 holds by construction), delays (<= dmax), sensor-loss
bursts (<= Nb consecutive), actuator-loss bursts (<= Na consecutive) and the switching
signal of the supervisor. The objective is the smallest normalised margin among
  m_Omega = 1 - max_k ||x_k||_P / ell          (state in certified set; Theorem 1(c))
  m_feas  = min_k (ell - f(Pi(c_k, r_k, K_R c_k))) / ell   (feasibility of fallback; Thm 1(a))
  m_info  = min_k (ell - f(c_k, r_k)) / ell    (information state in I; Thm 1(b))
  m_cont  = min_k min_i (r_ik - |x_ik - c_ik|) / rbar_i   (containment; Lemma 1(i))
  m_rad   = min_k min_i (rbar_i - r_ik) / rbar_i         (radius bound; Lemma 1(iii))
A negative value would be a counterexample.
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from bench_designs import make_design
from controllers import SafeSwitch

OUT = os.path.join(os.path.dirname(__file__), "results")
T = 300
START = float(os.environ.get('ADV_START', '0.99'))  # fraction of ell for f(c0, r0)
LARGE = float(os.environ.get('ADV_LARGE', '0'))    # >0: also detectable corruption of this size (monitored form)
SEG = 15  # number of piecewise-constant segments
_D = None


def _init(bench):
    global _D
    _D = make_design(bench)


def dim(n, m):
    # x0 angle(n) + dA signs(n*n) + dB(n*m) + w(p) + v(n) + a(n) + delay(n) + loss(n) + act(1) + mode(1), per segment
    return n + SEG * (n * n + n * m + 1 + n + n + n + n + 1 + 1)


def rollout(theta, return_trace=False):
    D = _D
    pl = D.plant; C = D.cert
    n, m = pl.n, pl.m
    th = np.clip(theta, -1, 1)
    i0 = 0
    dirv = th[:n] + 1e-9; i0 = n
    seg = th[i0:].reshape(SEG, -1)
    L = T // SEG
    # initial state: on the boundary f(x0, r0) = 0.99 ell
    r0 = np.full(n, D.vbar + D.abar)
    Phi = np.linalg.inv(C.Ph)
    c0 = Phi @ (dirv / np.linalg.norm(dirv))
    c0 *= max(START * D.ell - C.p @ r0, 0.0) / C.normP(c0)      # (c0, r0) on the boundary of I
    x0 = c0 + r0 * np.sign(seg[0, n * n + n * m + 1 + n:n * n + n * m + 1 + 2 * n])  # x0 anywhere in B(c0, r0)
    ctrl = SafeSwitch(D, "schedule", dwell=0, recovery=LARGE > 0)
    modes = np.repeat((seg[:, -1] > 0).astype(int), L)
    ctrl.mode_schedule = modes
    ctrl.reset(None, c0)
    arrivals = [[] for _ in range(T + 4 * D.dmax + 2)]
    x = x0.copy()
    u_app = np.zeros(m); buf = []
    run_s = np.zeros(n, int); run_a = 0
    m_omega = m_feas = m_info = m_cont = m_rad = np.inf
    n_alarm = 0
    modes = []
    onsets, xs = [], []
    for k in range(T):
        s = seg[k // L]
        o = 0
        dA = C.dA * np.sign(s[o:o + n * n]).reshape(n, n); o += n * n
        dB = C.dB * np.sign(s[o:o + n * m]).reshape(n, m); o += n * m
        w = np.atleast_1d(C.wbar * s[o:o + 1]); o += 1
        v = D.vbar * np.sign(s[o:o + n]); o += n
        sa = s[o:o + n]; o += n
        a = D.abar * np.sign(sa)
        if LARGE > 0:
            a = np.where(np.abs(sa) > 0.5, LARGE * np.sign(sa), a)   # detectable corruption (rejected)
        dly = np.round((s[o:o + n] + 1) / 2 * D.dmax).astype(int); o += n
        lossp = s[o:o + n] > 0; o += n
        actl = s[o] > 0; o += 1
        y = x + v + a
        for i in range(n):
            lost = lossp[i] and run_s[i] < D.Nb  # bursts of at most Nb consecutive losses
            run_s[i] = run_s[i] + 1 if lost else 0
            if not lost:
                arrivals[k + dly[i]].append((i, k, float(y[i])))
        u_cmd, newbuf, info = ctrl.step(k, arrivals[k], u_app)
        c, r = info["c"], info["r"]
        al_k = bool(info.get("alarm", False))
        n_alarm += int(al_k)
        if np.isfinite(info.get("tsafe", np.nan)):
            onsets.append((k, int(info["tsafe"]), bool(np.all(np.abs(x - c) <= r + 1e-9))))
        xs.append(x.copy()); modes.append(int(info.get("mode", 0)))
        # margins (evaluated on the controller's information state at k)
        if not (LARGE > 0 and (al_k or getattr(ctrl, "_alarm_seen", False))):   # monitored form: certified while monitor silent
            m_omega = min(m_omega, 1 - C.normP(x) / D.ell)
            m_info = min(m_info, (D.ell - C.f(c, r)) / D.ell)
            uf = np.atleast_1d(C.K @ c)
            _, rp = C.predict(c, r, uf)
            m_feas = min(m_feas, (D.ell - (C.normP(C.A @ c + C.B @ uf) + C.p @ rp)) / D.ell)
            m_cont = min(m_cont, float(np.min((r - np.abs(x - c) + 1e-9) / C.rbar)))
            m_rad = min(m_rad, float(np.min((C.rbar - r) / C.rbar)))
        ctrl._alarm_seen = getattr(ctrl, "_alarm_seen", False) or al_k
        al = actl and run_a < D.Na
        run_a = run_a + 1 if al else 0
        if not al or k == 0:
            u_app = np.clip(np.atleast_1d(u_cmd), -D.umax, D.umax)
            buf = list(newbuf) if newbuf is not None else []
        elif buf:
            u_app = np.clip(np.atleast_1d(buf.pop(0)), -D.umax, D.umax)
        x = (C.A + dA) @ x + (C.B + dB) @ u_app + C.E @ w
    # time-budget soundness at every alarm onset with containment (Proposition 2)
    xs = np.array(xs); sound = True; budgets = []
    for (k0, tb, cont0) in onsets:
        budgets.append(tb)
        if cont0:
            # Proposition 2(a): the budget holds while the recovery mode is active (modes 2/3)
            md = np.array(modes[k0:k0 + tb]); stop = np.where(md < 2)[0]
            end = k0 + (int(stop[0]) if len(stop) else len(md))
            v = np.where(np.any(np.abs(xs[k0:end]) > D.xmax, axis=1))[0]
            if len(v):
                sound = False
    mar = dict(omega=m_omega, feas=m_feas, info=m_info, cont=m_cont, rad=m_rad, alarms=n_alarm,
               budget_sound=float(sound), n_onsets=len(onsets))
    J = min(m_omega, m_feas, m_rad) + 1000 * min(0.0, m_info + 1e-9) + 1000 * min(0.0, m_cont)
    J -= 0.0 if sound else 1000.0
    return J, mar


def cem(bench="dcmotor", iters=25, pop=48, elite=8, seed=0, procs=2):
    D = make_design(bench)
    d = dim(D.plant.n, D.plant.m)
    rng = np.random.default_rng(seed)
    mu, sd = np.zeros(d), np.ones(d) * 0.8
    hist, best = [], (np.inf, None, None)
    n_eval = 0
    allmins = {k: np.inf for k in ("omega", "feas", "info", "cont", "rad")}
    tot_alarm = 0
    budget_ok, n_on = True, 0
    with Pool(procs, initializer=_init, initargs=(bench,)) as P:
        for it in range(iters):
            th = np.clip(mu + sd * rng.normal(size=(pop, d)), -1, 1)
            # bang-bang candidates: push half of the population to the box corners
            th[pop // 2:] = np.sign(th[pop // 2:] + 1e-9)
            res = P.map(rollout, list(th))
            n_eval += pop
            J = np.array([r[0] for r in res])
            for _, mar in res:
                tot_alarm += mar["alarms"]
                budget_ok = budget_ok and mar["budget_sound"] >= 1
                n_on += mar["n_onsets"]
                for k_ in allmins:
                    allmins[k_] = min(allmins[k_], mar[k_])
            idx = np.argsort(J)[:elite]
            mu = 0.7 * th[idx].mean(0) + 0.3 * mu
            sd = 0.7 * th[idx].std(0) + 0.3 * sd + 0.02
            if J[idx[0]] < best[0]:
                best = (float(J[idx[0]]), th[idx[0]].tolist(), res[idx[0]][1])
            hist.append(dict(it=it, best=float(J.min()), mean=float(J.mean())))
            print(it, round(J.min(), 4), {k_: round(float(v), 4) for k_, v in res[idx[0]][1].items()}, flush=True)
    return dict(bench=bench, n_eval=n_eval, best_objective=best[0], best_margins=best[2],
                min_margins_over_all=allmins, history=hist, counterexample=bool(best[0] < 0),
                monitor_alarm_steps=int(tot_alarm), budget_all_sound=bool(budget_ok), n_alarm_onsets=int(n_on))


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "dcmotor"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    t0 = time.time()
    r = cem(bench, iters=iters)
    r["wall_s"] = time.time() - t0
    r["start_fraction"] = START
    r["large"] = LARGE
    json.dump(r, open(os.path.join(OUT, f"adv_search_{bench}_start{START:.2f}" + (f"_large{LARGE:g}" if LARGE else "") + ".json"), "w"), indent=1, default=float)
    print(json.dumps({k: r[k] for k in ("n_eval", "best_objective", "best_margins", "min_margins_over_all",
                                        "counterexample")}, indent=1, default=float))
