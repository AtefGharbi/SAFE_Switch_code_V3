"""Result figures and tables (reads results/*.jsonl, *.json; writes figs/*.png, results/tables.json)."""
import json, os, sys, pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures import COL, INK, INK2, MUTED, GRID, ATTACK_SHADE, save, FIG, OUT

METHODS = ["SAFE-Switch", "Risk-threshold + filter", "Huang–Dong [1]", "LQR", "Robust H∞", "Robust MPC",
           "PPO w/o filter"]


def load(suite):
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"{suite}.jsonl"))]
    df = pd.DataFrame(rows)
    # average PPO-based methods over policy seeds -> one value per test episode
    num = [c for c in df.columns if c not in ("scenario", "method", "seed", "policy_seed")]
    g = df.groupby(["scenario", "method", "seed"], as_index=False)[num].mean()
    return df, g


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# ------------------------------------------------------------------ traces
def traces():
    from config import Design
    from experiments import scenarios_main, make_cfg, TEST_SEED0, _init
    import experiments as ex
    from scenarios import sample
    from sim import simulate
    ex._init()
    D = ex._D
    out = {}
    for scen, seed in [("E5 bias+net", 1), ("E2 mult (HD)", 0), ("E3 loss 0.5", 0)]:
        cfg = make_cfg(scen, scenarios_main()[scen], seed)
        sc = sample(cfg, TEST_SEED0 + seed, xmax=D.xmax)
        out[scen] = {"sc": {k: sc[k] for k in ("attack_on", "s_lost", "s_delay", "a_lost", "attack_chans")}}
        for m in ["SAFE-Switch@0", "Risk-threshold + filter", "Huang–Dong [1]", "LQR", "Robust H∞",
                  "Robust MPC", "PPO w/o filter@0"]:
            ctrl = ex.build(m)
            kf = []
            if hasattr(ctrl, "kf") or m.startswith("SAFE") or m.startswith("PPO"):
                orig = ctrl.step

                def st(k, p, u, _o=orig, _c=ctrl, _l=kf):
                    r = _o(k, p, u)
                    _l.append(_c.kf.x.copy() if hasattr(_c, "kf") else np.full(2, np.nan))
                    return r
                ctrl.step = st
            o = simulate(D, ctrl, sc)
            o["KF"] = np.array(kf) if kf else None
            out[scen][m.split("@")[0]] = o
    with open(os.path.join(OUT, "traces.pkl"), "wb") as f:
        pickle.dump(out, f)


def _shade(ax, att, t):
    on = False
    for k in range(len(att)):
        if att[k] and not on:
            k0 = k; on = True
        if on and (not att[k] or k == len(att) - 1):
            ax.axvspan(t[k0], t[k], color=ATTACK_SHADE, lw=0, zorder=0)
            on = False


def fig_mechanism():
    tr = pickle.load(open(os.path.join(OUT, "traces.pkl"), "rb"))
    scen = "E5 bias+net"
    o = tr[scen]["SAFE-Switch"]; sc = tr[scen]["sc"]
    T = len(o["U"]); t = np.arange(T) * 0.05
    fig, axs = plt.subplots(4, 1, figsize=(7.0, 5.6), sharex=True, gridspec_kw=dict(hspace=0.25))
    ax = axs[0]
    _shade(ax, sc["attack_on"], t)
    for i, nm in enumerate(["current ch.", "speed ch."]):
        lost = np.where(sc["s_lost"][:T, i])[0]
        ax.plot(t[lost], np.full(len(lost), 1.6 - i), "|", color=COL["Robust H∞"], ms=4, mew=0.8)
        dl = np.where(sc["s_delay"][:T, i] > 0)[0]
        ax.plot(t[dl], np.full(len(dl), 1.3 - i), "|", color=COL["PPO w/o filter"], ms=3, mew=0.5)
    al = np.where(sc["a_lost"][:T])[0]
    ax.plot(t[al], np.full(len(al), -0.2), "|", color=COL["Robust MPC"], ms=4, mew=0.8)
    ax.set_yticks([1.45, 0.45, -0.2]); ax.set_yticklabels(["current", "speed", "actuator"], fontsize=6.5)
    ax.set_ylim(-0.5, 1.9); ax.grid(False)
    ax.set_title("(a) network and attack timeline\n(shaded: bias attack on the " +
                 ("speed" if sc["attack_chans"][1] else "current") + " channel)", loc="left")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COL["Robust H∞"], marker="|", ls="", ms=6, label="sensor packet lost"),
          Line2D([], [], color=COL["PPO w/o filter"], marker="|", ls="", ms=6, label="sensor packet delayed"),
          Line2D([], [], color=COL["Robust MPC"], marker="|", ls="", ms=6, label="actuator packet lost")]
    ax.legend(handles=hs, fontsize=6.5, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.08))
    ax = axs[1]
    _shade(ax, sc["attack_on"], t)
    ax.plot(t, o["RISK"], color=COL["SAFE-Switch"])
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_ylabel("risk $\\rho_k$"); ax.set_ylim(0, 1); ax.set_title("(b) calibrated attack risk", loc="left")
    ax = axs[2]
    _shade(ax, sc["attack_on"], t)
    ax.step(t, o["MODE"], where="post", color=COL["SAFE-Switch"], label="SAFE-Switch (PPO)")
    th = tr[scen]["Risk-threshold + filter"]
    ax.step(t, th["MODE"] * 0.9 + 0.05, where="post", color=COL["Risk-threshold + filter"], lw=1.0,
            label="risk-threshold rule")
    iv = np.where(o["INTERV"])[0]
    ax.plot(t[iv], np.full(len(iv), 0.5), ".", color=COL["Huang–Dong [1]"], ms=1.5, label="filter intervention")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["N", "R"]); ax.set_title("(c) selected mode", loc="left")
    ax.legend(fontsize=6.5, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.32))
    ax = axs[3]
    _shade(ax, sc["attack_on"], t)
    for m in ["LQR", "Huang–Dong [1]", "SAFE-Switch"]:
        X = tr[scen][m]["X"][:-1]
        ax.plot(t, np.linalg.norm(X, axis=1), color=COL[m], label=m, lw=1.2 if m == "SAFE-Switch" else 0.9)
    ax.set_ylabel("$\\|x_k\\|$"); ax.set_xlabel("time (s)"); ax.set_title("(d) tracking error", loc="left")
    ax.legend(fontsize=6.5, ncol=3, loc="upper right")
    save(fig, "fig5_mechanism.png")


def fig_estimation():
    tr = pickle.load(open(os.path.join(OUT, "traces.pkl"), "rb"))
    scen = "E5 bias+net"
    o = tr[scen]["SAFE-Switch"]; sc = tr[scen]["sc"]
    T = len(o["U"]); t = np.arange(T) * 0.05
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True, gridspec_kw=dict(hspace=0.3))
    for i, nm in enumerate(["current error $x_1$", "speed error $x_2$"]):
        ax = axs[i]
        _shade(ax, sc["attack_on"], t)
        c, r = o["C"][:, i], o["R"][:, i]
        ax.fill_between(t, c - r, c + r, color=COL["SAFE-Switch"], alpha=0.18, lw=0, label="set $c_k\\pm r_k$")
        ax.plot(t, o["X"][:-1, i], color=INK, lw=0.9, label="true state")
        ax.plot(t, o["KF"][:, i], color=COL["Huang–Dong [1]"], lw=0.8, label="Kalman estimate")
        ax.set_ylabel(nm)
        if i == 0:
            ax.legend(fontsize=6.5, ncol=3, loc="upper right")
    axs[1].set_xlabel("time (s)")
    lo, hi = np.percentile(o["X"][:-1, 1], [1, 99])
    axs[1].set_ylim(min(lo, -1.0), max(hi, 1.5))
    save(fig, "fig6_estimation.png")


def fig_filter():
    tr = pickle.load(open(os.path.join(OUT, "traces.pkl"), "rb"))
    scen = "E2 mult (HD)"
    o = tr[scen]["SAFE-Switch"]; sc = tr[scen]["sc"]
    T = len(o["U"]); t = np.arange(T) * 0.05
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    _shade(ax, sc["attack_on"], t)
    ax.plot(t, o["UC"], color=COL["Huang–Dong [1]"], lw=0.8, label="candidate $u^c_k$")
    ax.plot(t, o["U"], color=COL["SAFE-Switch"], lw=1.0, label="applied $u_k$")
    iv = np.where(o["INTERV"])[0]
    ax.plot(t[iv], np.full(len(iv), ax.get_ylim()[0]), "|", color=COL["Risk-threshold + filter"], ms=4,
            label="intervention")
    ax.set_xlabel("time (s)"); ax.set_ylabel("input")
    ax.legend(fontsize=6.5, ncol=3, loc="upper right")
    save(fig, "fig8_filter.png")


def fig_trajectories():
    tr = pickle.load(open(os.path.join(OUT, "traces.pkl"), "rb"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, scen in zip(axs, ["E2 mult (HD)", "E5 bias+net"]):
        sc = tr[scen]["sc"]
        T = len(sc["attack_on"]); t = np.arange(T) * 0.05
        _shade(ax, sc["attack_on"], t)
        for m in METHODS:
            if m not in tr[scen]:
                continue
            X = tr[scen][m]["X"][:-1, 1]
            ax.plot(t, np.clip(X, -6, 6), color=COL[m], lw=1.2 if m == "SAFE-Switch" else 0.8, label=m)
        ax.axhline(4, color=MUTED, ls="--", lw=0.8); ax.axhline(-4, color=MUTED, ls="--", lw=0.8)
        ax.set_ylim(-6, 6); ax.set_xlabel("time (s)"); ax.set_title(scen, loc="left")
    axs[0].set_ylabel("speed error $x_2$ (clipped ±6)")
    axs[1].legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig7_trajectories.png")


# ------------------------------------------------------------------ sensitivity
def fig_sensitivity(g):
    fams = [("bias magnitude", [f"E2 bias {a}" for a in [0.1, 0.3, 1.0, 2.0]], [0.1, 0.3, 1.0, 2.0]),
            ("sensor loss probability", ["E1 nominal"] + [f"E3 loss {p}" for p in [0.1, 0.3, 0.5]], [0, 0.1, 0.3, 0.5]),
            ("max delay $d_{max}$ (steps)", ["E1 nominal"] + [f"E4 delay {d}" for d in [1, 3, 6, 10]], [0, 1, 3, 6, 10]),
            ("parameter uncertainty (±)", [f"E7 unc {u}" for u in [0.0, 0.03, 0.1, 0.2, 0.3]], [0, 0.03, 0.1, 0.2, 0.3])]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.45))
    for ax, (lab, scs, xs) in zip(axs, fams):
        for m in METHODS:
            ys, lo, hi = [], [], []
            for s in scs:
                v = g[(g.scenario == s) & (g.method == m)]["rmse"].values
                if len(v) == 0:
                    ys.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                med = np.median(v)
                ys.append(med); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
            ax.plot(xs, ys, "o-", color=COL[m], ms=3, lw=1.2 if m == "SAFE-Switch" else 0.9, label=m)
            ax.fill_between(xs, lo, hi, color=COL[m], alpha=0.08, lw=0)
        ax.set_yscale("log"); ax.set_xlabel(lab, fontsize=7)
        from matplotlib.ticker import LogLocator, FormatStrFormatter, NullFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g")); ax.yaxis.set_minor_formatter(NullFormatter())
    axs[0].set_ylabel("RMSE of $\\|x\\|$ (median, IQR)")
    axs[-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "fig9_sensitivity.png")


def fig_learning():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.4))
    ax = axs[0]
    for s in range(5):
        d = pickle.load(open(os.path.join(OUT, f"ppo_full_seed{s}.pkl"), "rb"))
        r = np.array([c["ret"] for c in d["curve"]])
        med = pd.Series(r).rolling(5, min_periods=1).median()
        ax.plot(np.arange(1, len(r) + 1), -med, color=COL["SAFE-Switch"], alpha=0.35 + 0.13 * s, lw=0.9)
    ax.set_yscale("log"); ax.set_xlabel("PPO update"); ax.set_ylabel("training episode cost (rolling median)")
    ax.set_title("(a) training cost, 5 seeds", loc="left")
    ax = axs[1]
    p = os.path.join(OUT, "curve_eval.json")
    if os.path.exists(p):
        c = json.load(open(p))
        for key, colr, lab in [("warm", COL["SAFE-Switch"], "PPO, BC warm start"),
                               ("cold", COL["PPO w/o filter"], "PPO, random init")]:
            h = c[key]
            ax.plot([x["it"] for x in h], [x["val"] for x in h], "o-", ms=2.5, color=colr, label=lab)
        ax.axhline(c["threshold"], color=COL["Risk-threshold + filter"], ls="--", lw=0.9, label="risk-threshold rule")
        ax.axhline(c["always_R"], color=MUTED, ls=":", lw=0.9, label="always R")
        ax.set_xlabel("PPO update"); ax.set_ylabel("validation cost (8 fixed episodes)")
        ax.set_title("(b) fixed-validation convergence", loc="left")
        ax.legend(fontsize=6)
    save(fig, "fig10_learning.png")


def fig_filter_stress():
    import experiments as ex
    from experiments import scenarios_stress, make_cfg, TEST_SEED0
    from scenarios import sample
    from sim import simulate
    ex._init(); D = ex._D
    scen = "E10 burst-bias 6 speed"
    cfg = make_cfg(scen, scenarios_stress()[scen], 0)
    sc = sample(cfg, TEST_SEED0 + 0, xmax=D.xmax)
    of = simulate(D, ex.build("Filter only (N)"), sc)
    on = simulate(D, ex.build("Always N w/o filter"), sc)
    T = len(of["U"]); t = np.arange(T) * 0.05
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True, gridspec_kw=dict(hspace=0.3))
    ax = axs[0]
    _shade(ax, sc["attack_on"], t)
    ax.plot(t, of["UC"], color=COL["Huang–Dong [1]"], lw=0.8, label="candidate $u^c_k=K_N\\hat x_k$")
    ax.plot(t, of["U"], color=COL["SAFE-Switch"], lw=1.0, label="filtered $u_k$")
    iv = np.where(of["INTERV"])[0]
    ax.plot(t[iv], np.full(len(iv), -10.5), "|", color=COL["Risk-threshold + filter"], ms=4, label="intervention")
    ax.set_ylabel("input"); ax.set_ylim(-11, 11)
    ax.set_title("(a) worst-case supervisor (always N) with the safety filter", loc="left")
    ax.legend(fontsize=6.5, ncol=3, loc="upper right")
    ax = axs[1]
    _shade(ax, sc["attack_on"], t)
    ax.plot(t, on["X"][:-1, 1], color=COL["LQR"], lw=0.9, label="always N, no filter")
    ax.plot(t, of["X"][:-1, 1], color=COL["SAFE-Switch"], lw=1.1, label="always N + safety filter")
    ax.axhline(4, color=MUTED, ls="--", lw=0.8); ax.axhline(-4, color=MUTED, ls="--", lw=0.8)
    ax.set_ylabel("speed error $x_2$"); ax.set_xlabel("time (s)")
    ax.set_title("(b) state: constraint $|x_2|\\leq 4$ dashed", loc="left")
    ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    save(fig, "fig8_filter.png")


def fig_tank():
    import experiments_v2 as e
    from scenarios import sample
    from sim import simulate
    S = e.scen_dict("tank")
    D = e.design("tank")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, scen, seed in zip(axs, ["E5 bias+net", "E2 mult (HD)"], [1, 0]):
        cfg = e.make_cfg(scen, S[scen], seed)
        sc = sample(cfg, e.TEST_SEED0 + seed, xmax=D.xmax)
        T = cfg.T; t = np.arange(T) * D.Ts
        _shade(ax, sc["attack_on"], t)
        for m, lab, col in [("LQR", "LQR", COL["LQR"]), ("Huang–Dong [1]", "Huang–Dong [1]", COL["Huang–Dong [1]"]),
                            ("Always N w/o filter", "always N, no filter", COL["PPO w/o filter"]),
                            ("MLP-PPO@0", "SAFE-Switch (PPO)", COL["SAFE-Switch"])]:
            ctrl, M = e.build(m, "tank", {})
            o = simulate(D, ctrl, sc)
            ax.plot(t, np.clip(o["X"][:-1, 1], -8, 8), color=col, lw=1.2 if "SAFE" in lab else 0.8, label=lab)
        ax.axhline(6, color=MUTED, ls="--", lw=0.8); ax.axhline(-6, color=MUTED, ls="--", lw=0.8)
        ax.set_ylim(-8, 8); ax.set_xlabel("time (s)"); ax.set_title(f"coupled tanks, {scen}" + (" (LQR ≡ always N)" if "mult" in scen else ""), loc="left")
    axs[0].set_ylabel("lower-tank level deviation (cm)")
    axs[1].legend(fontsize=6, loc="lower right")
    save(fig, "fig_tank.png")


if __name__ == "__main__":
    what = sys.argv[1:]
    if "traces" in what:
        traces()
    if "mech" in what:
        fig_mechanism(); fig_estimation(); fig_trajectories()
    if "sens" in what:
        _, g = load("main"); fig_sensitivity(g)
    if "fstress" in what:
        fig_filter_stress()
    if "learn" in what:
        fig_learning()
    if "tank" in what:
        fig_tank()
