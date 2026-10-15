"""V2 tables, statistics and figures -> results/tables_v2.json, figs/*.png"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures import COL, INK, INK2, MUTED, save, FIG
from figs_results import holm

OUT = os.path.join(os.path.dirname(__file__), "results")
rng = np.random.default_rng(0)


def jl(name):
    rows = []
    for l in open(os.path.join(OUT, name)):
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    return pd.DataFrame(rows)


def big(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.3f}")


def cp_upper(n, k=0, conf=0.95):
    """Exact one-sided Clopper-Pearson upper bound."""
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(conf, k + 1, n - k))


def tci(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    m = d.mean(); h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return m, m - h, m + h


def boot2(df, col, B=2000):
    """Two-level bootstrap of the mean: resample policy seeds, then test seeds."""
    ps = df.policy_seed.unique(); ts = df.seed.unique()
    piv = df.pivot_table(index="policy_seed", columns="seed", values=col, aggfunc="mean").values
    means = []
    for _ in range(B):
        a = rng.integers(0, len(ps), len(ps)); b = rng.integers(0, len(ts), len(ts))
        means.append(np.nanmean(piv[np.ix_(a, b)]))
    return float(np.nanmean(piv)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ main-suite statistics upgrade
def main_stats(T):
    df = jl("main.jsonl")
    g = df.groupby(["scenario", "method", "seed"], as_index=False).mean(numeric_only=True)
    inenv = ["E1 nominal", "E2 stealth 0.03", "E3 loss 0.1", "E3 loss 0.3", "E3 loss 0.5", "E3 actuator loss 0.3",
             "E4 delay 1", "E4 delay 3"]
    n_ep = int(len(g[(g.method == "SAFE-Switch") & g.scenario.isin(inenv)]))
    n_runs = int(len(df[(df.method == "SAFE-Switch") & df.scenario.isin(inenv)]))
    T["inenv_zero"] = dict(n_episodes=n_ep, n_runs=n_runs, cp95_episode=cp_upper(n_ep), cp95_run=cp_upper(n_runs))
    rows = ""
    T["paired"] = {}
    meths = ["Risk-threshold + filter", "Robust H∞", "Robust MPC", "LQR", "Huang–Dong [1]"]
    for sc in ["E5 bias+net", "E2 bias 1.0", "E3 loss 0.5", "E1 nominal"]:
        a = g[(g.scenario == sc) & (g.method == "SAFE-Switch")].sort_values("seed")
        for m in meths:
            b = g[(g.scenario == sc) & (g.method == m)].sort_values("seed")
            d1 = tci(a.rmse.values - b.rmse.values); d2 = tci(a.energy.values - b.energy.values)
            T["paired"][f"{sc}|{m}"] = dict(rmse=[float(x) for x in d1], energy=[float(x) for x in d2])
            rows += (f"| {sc} | {m} | {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}] | "
                     f"{d2[0]:+.2f} [{d2[1]:+.2f}, {d2[2]:+.2f}] |\n")
    T["TABLE_PAIRED"] = (": TABLE IX. Paired differences (SAFE-Switch − baseline) of RMSE and control energy with 95 % "
                         "t-intervals over 20 paired test seeds (DC motor, main suite). Negative = SAFE-Switch lower.\n\n"
                         "| Scenario | baseline | ΔRMSE [95 % CI] | Δenergy [95 % CI] |\n|---|---|---|---|\n" + rows)
    # policy-seed level vs scenario level for SAFE-Switch (5 policies)
    lv = ""
    for sc in ["E1 nominal", "E2 mult (HD)", "E5 bias+net"]:
        d = df[(df.scenario == sc) & (df.method == "SAFE-Switch")]
        pm = d.groupby("policy_seed").rmse.mean()
        m, lo, hi = boot2(d, "rmse")
        T.setdefault("levels", {})[sc] = dict(mean=float(m), lo=float(lo), hi=float(hi), sd_policy=float(pm.std(ddof=1)),
                                              sd_test=float(d.groupby('seed').rmse.mean().std(ddof=1)))
        lv += (f"| {sc} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {pm.std(ddof=1):.4f} | "
               f"{d.groupby('seed').rmse.mean().std(ddof=1):.3f} |\n")
    T["TABLE_LEVELS"] = (": TABLE X. Two experimental levels for the main SAFE-Switch results (5 policy seeds × 20 test "
                         "seeds): two-level bootstrap interval of the mean RMSE and standard deviations of policy-seed "
                         "means and test-seed means.\n\n| Scenario | mean RMSE | 95 % two-level CI | SD over policy seeds | "
                         "SD over test seeds |\n|---|---|---|---|---|\n" + lv)


# ------------------------------------------------------------------ E11 adversarial search
def adv(T):
    r = {}
    for s in ["0.99", "0.00", "0.99_large2"]:
        for b in ["dcmotor", "tank"]:
            p = os.path.join(OUT, f"adv_search_{b}_start{s}.json")
            if os.path.exists(p):
                r[(b, s)] = json.load(open(p))
    rows = ""
    for (b, s), d in r.items():
        m = d["min_margins_over_all"]
        lab = (f"{float(s[:4]):.2f}ℓ" + ("‡" if "large" in s else ""))
        extra = ""
        rows += (f"| {'DC motor' if b == 'dcmotor' else 'coupled tanks'} | {lab}{extra} | {d['n_eval']} | "
                 f"{m['omega']:.4f} | {m['feas']:.4f} | {max(m['info'], 0):.4f} | {max(m['cont'], 0):.2e} | "
                 f"{m['rad']:.4f} | {'none' if not d['counterexample'] else 'FOUND'} |\n")
    T["TABLE_ADV"] = (": TABLE XI. Certificate validation by adversarial search (E11, cross-entropy method over initial "
                      "state, time-varying ΔA/ΔB, disturbance, noise, accepted corruption |a| ≤ ā, delays, worst-case "
                      "loss bursts and switching signal; 300-step episodes). Smallest normalised margins found "
                      "(negative = counterexample). ‡ = with additional detectable corruption of magnitude 2 (monitored form, "
                      "Corollary 1); margins evaluated while the monitor is silent, and the recovery budget of "
                      "Proposition 2 checked at every alarm onset: "
                      + "; ".join(f"{'motor' if b == 'dcmotor' else 'tanks'}: {d.get('n_alarm_onsets', 0)} onsets, budget "
                                  f"{'always respected' if d.get('budget_all_sound') else 'VIOLATED'}"
                                  for (b, s), d in r.items() if "large" in s) + ".\n\n| Plant | start f(c₀,r₀) | episodes | Ω margin | fallback-"
                      "feasibility margin | information-set margin | containment margin | radius margin | "
                      "counterexample |\n|---|---|---|---|---|---|---|---|---|\n" + rows)
    T["adv"] = {f"{b}_{s}": d["min_margins_over_all"] | dict(n=d["n_eval"], cex=d["counterexample"],
                                                         alarms=d.get("monitor_alarm_steps"), onsets=d.get("n_alarm_onsets"),
                                                         sound=d.get("budget_all_sound"))
                for (b, s), d in r.items()}


# ------------------------------------------------------------------ certificates of both plants
def certs(T):
    rows = ""
    T["cert_v2"] = {}
    for b, name in [("dcmotor", "DC motor"), ("tank", "coupled tanks")]:
        p = os.path.join(OUT, f"cert_verification_{b}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); c = d["conditions"]
        T["cert_v2"][b] = d
        rows += (f"| {name} | {d['rho']:.4f} | {d['alpha']:.4f} | {d['beta']:.4f} | {d['c0']:.4f} | "
                 f"[{d['rbar'][0]:.3f}, {d['rbar'][1]:.3f}] | {d['qbar']:.3f} | {d['ell']:.3f} | "
                 f"{c['C2_level_invariance']['lhs']:.4f} ({c['C2_level_invariance']['margin']:.4f}) | "
                 f"{d['sampled_onestep_worst_slack']:.4f} | {'pass' if d['all_pass'] else 'FAIL'} |\n")
    T["TABLE_CERT"] = (": TABLE III. Certificate quantities and conditions (machine-readable files "
                       "`cert_verification_<plant>.json`). C2 column: left-hand side (margin to ℓ). Sampled slack: "
                       "largest value of f(Π(c,r,K_Rc)) − ℓ over 2×10⁵ sampled (c,r) ∈ I (must be ≤ 0).\n\n"
                       "| Plant | ρ | α (C1) | β | c₀ | r̄ | q̄ (C3) | ℓ | C2 LHS (margin) | sampled slack | (C1)–(C5) |\n"
                       "|---|---|---|---|---|---|---|---|---|---|---|\n" + rows)


# ------------------------------------------------------------------ E12 supervisors
def e12(T):
    df = jl("e12.jsonl")
    order = ["Threshold + filter", "Tuned threshold + filter", "Fixed R + filter", "Always N + filter",
             "MLP-PPO", "GRU-PPO", "CVaR-PPO"]
    sc_order = list(dict.fromkeys(df.scenario))
    hdr = "| Supervisor (all with filter) | seeds | " + " | ".join(sc_order) + " | mean | switches | interv. % |\n"
    hdr += "|---|---|" + "---|" * (len(sc_order) + 3) + "\n"
    rows = ""
    T["e12"] = {}
    ref = df[df.method == "Tuned threshold + filter"].groupby(["scenario", "seed"]).rmse.mean()
    for m in order:
        d = df[df.method == m]
        if not len(d):
            continue
        ns = d.policy_seed.nunique() if d.policy_seed.max() >= 0 else 0
        cells = []
        for sc in sc_order:
            v = d[d.scenario == sc].groupby("seed").rmse.mean()
            cells.append(big(v.mean()))
        allm = d.groupby(["scenario", "seed"]).rmse.mean()
        rows += (f"| {m} | {ns if ns else '—'} | " + " | ".join(cells) +
                 f" | {allm.mean():.3f} | {d.switches.mean():.1f} | {100 * d.interv.mean():.1f} |\n")
        T["e12"][m] = dict(mean=float(allm.mean()), per={sc: float(d[d.scenario == sc].rmse.mean()) for sc in sc_order},
                           switches=float(d.switches.mean()), interv=float(d.interv.mean()),
                           viol=float(d.viol_any.mean()))
    T["TABLE_E12"] = (": TABLE XII. Supervisor study (E12, DC motor): RMSE of ‖x‖ per scenario family (mean over 20 "
                      "test seeds; for PPO variants first averaged over policy seeds), identical estimator, filter and "
                      "test trajectories.\n\n" + hdr + rows)
    # paired comparison vs tuned threshold, two levels
    rows = ""
    T["e12_vs_tuned"] = {}
    for m in ["Threshold + filter", "Fixed R + filter", "MLP-PPO", "GRU-PPO", "CVaR-PPO"]:
        d = df[df.method == m]
        if not len(d):
            continue
        a = d.groupby(["scenario", "seed"]).rmse.mean()
        diff = (a - ref.reindex(a.index)).values
        mm, lo, hi = tci(diff)
        ps, wins, losses = [], 0, 0
        for sc in sc_order:
            x = a.xs(sc).sort_index().values; y = ref.xs(sc).sort_index().values
            dd = x - y
            p = stats.wilcoxon(x, y, zero_method="zsplit").pvalue if np.any(np.abs(dd) > 1e-12) else 1.0
            ps.append(p)
        pa = holm(np.array(ps))
        meds = [np.median(a.xs(sc).sort_index().values - ref.xs(sc).sort_index().values) for sc in sc_order]
        wins = int(sum((pa < 0.05) & (np.array(meds) < 0))); losses = int(sum((pa < 0.05) & (np.array(meds) > 0)))
        # policy-seed level: spread of per-policy mean differences
        if d.policy_seed.max() >= 0:
            pdiff = []
            for s_, ds in d.groupby("policy_seed"):
                b = ds.groupby(["scenario", "seed"]).rmse.mean()
                pdiff.append(float((b - ref.reindex(b.index)).mean()))
            spread = f"{np.min(pdiff):+.4f} … {np.max(pdiff):+.4f}"
            better_seeds = int(sum(np.array(pdiff) < 0))
            nseeds = len(pdiff)
        else:
            spread, better_seeds, nseeds = "—", None, None
        rows += (f"| {m} | {mm:+.4f} [{lo:+.4f}, {hi:+.4f}] | {wins} / {len(sc_order) - wins - losses} / {losses} | "
                 f"{spread} | {'—' if nseeds is None else f'{better_seeds}/{nseeds}'} |\n")
        T["e12_vs_tuned"][m] = dict(mean=mm, lo=lo, hi=hi, wins=wins, losses=losses, better_seeds=better_seeds,
                                    nseeds=nseeds)
    T["TABLE_E12B"] = (": TABLE XIII. Paired comparison with the tuned threshold rule (E12). ΔRMSE pooled over families "
                       "and test seeds with 95 % CI; per-family Wilcoxon (Holm) better/no difference/worse; range of the "
                       "mean difference across individual policy seeds; number of policy seeds better than the tuned "
                       "rule.\n\n| Supervisor | ΔRMSE vs tuned rule | families better / n.s. / worse | per-policy-seed "
                       "range | seeds better |\n|---|---|---|---|---|\n" + rows)


# ------------------------------------------------------------------ E15 hardening
def e15(T):
    df = jl("e15.jsonl")
    order = ["Threshold + filter", "Threshold + PR", "Threshold + Auth2", "Threshold + Auth3", "Threshold + Auth5",
             "Threshold + Auth3 PR", "Fixed R + filter", "LQR", "Robust H∞", "Robust MPC", "Huang–Dong [1]"]
    sc_order = list(dict.fromkeys(df.scenario))
    hdr = "| Variant | " + " | ".join(sc_order) + " |\n|---|" + "---|" * len(sc_order) + "\n"
    rows = ""
    T["e15"] = {}
    for m in order:
        d = df[df.method == m]
        if not len(d):
            continue
        cells = []
        for sc in sc_order:
            v = d[d.scenario == sc]
            if not len(v):
                cells.append("—"); continue
            cont = f", {100 * v.contain.mean():.0f} %" if m.startswith("Threshold") else ""
            cells.append(f"{big(v.rmse.mean())} ({100 * v.viol_any.mean():.0f}{cont})")
        rows += f"| {m} | " + " | ".join(cells) + " |\n"
        T["e15"][m] = {sc: dict(rmse=float(d[d.scenario == sc].rmse.mean()), viol=float(d[d.scenario == sc].viol_any.mean()),
                                contain=float(d[d.scenario == sc].contain.mean()),
                                alarm=float(d[d.scenario == sc].alarm_frac.mean()) if "alarm_frac" in d else np.nan,
                                budget_min=float(np.nanmin(d[d.scenario == sc].budget_min_s)) if "budget_min_s" in d and d[d.scenario == sc].budget_min_s.notna().any() else np.nan,
                                sound=float(d[d.scenario == sc].budget_sound.min()) if "budget_sound" in d and d[d.scenario == sc].budget_sound.notna().any() else np.nan)
                       for sc in sc_order if len(d[d.scenario == sc])}
    T["TABLE_E15"] = (": TABLE XIV. Threat-model hardening (E15, DC motor, risk-threshold supervisor): RMSE (violation-"
                      "episode %, containment %) over 20 seeds. PR = persistent-attack rule + recovery mode with "
                      "certified time budget; AuthM = set estimator fuses only authenticated packets (every M-th).\n\n"
                      + hdr + rows)
    # certified budgets are meaningful only where the recovery mode runs (PR variants)
    d = df[df.method.str.contains("PR")]
    if "budget_sound" in d and len(d):
        T["e15_budget"] = dict(min_budget_s=float(np.nanmin(d.budget_min_s)), all_sound=bool(d.budget_sound.min() >= 1),
                               n_onsets=int(d.n_alarm.sum()), median_mean_budget_s=float(np.nanmedian(d.budget_mean_s)),
                               frac_recovery=float(d.frac_recovery.mean()) if "frac_recovery" in d else None,
                               frac_hold_of_recovery=float(d.frac_hold.sum() / max(d.frac_recovery.sum(), 1e-12))
                               if "frac_hold" in d else None)


# ------------------------------------------------------------------ tank
def tank(T):
    df = jl("tank.jsonl")
    order = ["MLP-PPO", "Threshold + filter", "Fixed R + filter", "Always N + filter", "Always N w/o filter",
             "Threshold + PR", "Threshold + Auth3", "LQR", "Robust H∞", "Robust MPC", "Huang–Dong [1]"]
    g = df.groupby(["scenario", "method", "seed"], as_index=False).mean(numeric_only=True)
    sc_order = list(dict.fromkeys(df.scenario))
    hdr = "| Method | " + " | ".join(sc_order) + " |\n|---|" + "---|" * len(sc_order) + "\n"
    rows = ""
    T["tank"] = {}
    for m in order:
        d = g[g.method == m]
        if not len(d):
            continue
        cells = [f"{big(d[d.scenario == sc].rmse.mean())} ({100 * d[d.scenario == sc].viol_any.mean():.0f})"
                 for sc in sc_order]
        rows += f"| {m}{' (3 seeds)' if m == 'MLP-PPO' else ''} | " + " | ".join(cells) + " |\n"
        T["tank"][m] = {sc: dict(rmse=float(d[d.scenario == sc].rmse.mean()), viol=float(d[d.scenario == sc].viol_any.mean()),
                                 contain=float(d[d.scenario == sc].contain.mean()), infeas=float(d[d.scenario == sc].infeas.mean()))
                        for sc in sc_order}
    inenv = ["E1 nominal", "E3 loss 0.5", "E4 delay 3"]
    d = df[(df.method.isin(["MLP-PPO", "Threshold + filter", "Fixed R + filter", "Always N + filter"])) &
           (df.scenario.isin(inenv))]
    T["tank_inenv"] = dict(n=int(len(d)), viol=int(d.viol_any.sum()), infeas=float(d.infeas.max()),
                           contain_min=float(d.contain.min()), cp95=cp_upper(int(len(d))))
    T["TABLE_TANK"] = (": TABLE VIII. Second certified benchmark (coupled tanks, 20 paired seeds): RMSE of ‖x‖ in cm "
                       "(violation-episode %).\n\n" + hdr + rows)


# ------------------------------------------------------------------ E13 Pareto
def e13(T):
    df = jl("e13.jsonl")
    df = df[~df.dkw.str.contains('"lam": 0.9044')]     # superseded: 0.9044 < rho by 2e-5 (C5); rerun with 0.9045
    from bench_designs import make_design
    from verify_certificate import verify
    marg = {}
    for dk in df.dkw.unique():
        kw = json.loads(dk)
        D = make_design("dcmotor", **kw)
        C = D.cert
        lhs = C.alpha * D.ell + max(C.beta - C.alpha, 0) * C.qbar + C.c0
        marg[dk] = dict(margin=float(D.ell - lhs), ok=bool(D.cert_ok), lam=kw["lam"], abar=kw["abar"])
    df["cert_margin"] = df.dkw.map(lambda k: marg[k]["margin"])
    # the a-posteriori bound is reported only where (13) was checked over the second half of the episode
    df.loc[df.thm2_frac_checked < 0.999, "thm2_bound_tail"] = np.nan
    df["lam"] = df.dkw.map(lambda k: marg[k]["lam"]); df["abar"] = df.dkw.map(lambda k: marg[k]["abar"])
    agg = df.groupby(["method", "lam", "abar", "scenario"]).agg(rmse=("rmse", "mean"), energy=("energy", "mean"),
                                                                 interv=("interv", "mean"),
                                                                 thm2=("thm2_holds", "min"),
                                                                 bound=("thm2_bound_tail", "mean"),
                                                                 margin=("cert_margin", "first")).reset_index()
    T["e13"] = agg.to_dict(orient="records")
    T["e13_thm2_all_hold"] = bool(df.thm2_holds.min() >= 1)
    T["e13_thm2_by_scen"] = df.groupby("scenario").thm2_holds.mean().to_dict()
    T["e13_thm2_checked"] = df.groupby("scenario").thm2_frac_checked.mean().to_dict() if "thm2_frac_checked" in df else {}
    T["e13_margins"] = {k: v for k, v in marg.items()}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.45))
    for ax, sc in zip(axs, ["E1 nominal", "E5 bias+net", "E2 mult (HD)"]):
        for m, col in [("Tuned threshold + filter", COL["Risk-threshold + filter"]), ("Always N + filter", COL["LQR"])]:
            for ab, mk in [(0.03, "o"), (0.06, "s")]:
                a = agg[(agg.method == m) & (agg.scenario == sc) & (agg.abar == ab)].sort_values("lam")
                ax.plot(100 * a.interv, a.rmse, mk + "-", color=col, ms=3.5, lw=0.9, alpha=1 if ab == 0.03 else 0.55,
                        label=f"{'tuned rule' if 'Tuned' in m else 'always N'}, ā={ab}")
                for _, r in a.iterrows():
                    if ab == 0.03 and m.startswith("Tuned") and sc == "E1 nominal":
                        ax.annotate(f"λ={r.lam:.2f}", (100 * r.interv, r.rmse), fontsize=5, color=INK2,
                                    xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("filter interventions (%)"); ax.set_title(sc, loc="left")
    axs[0].set_ylabel("RMSE of ‖x‖")
    axs[-1].legend(fontsize=5.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "fig_pareto.png")
    rows = ""
    for dk, v in sorted(marg.items(), key=lambda kv: (kv[1]["abar"], kv[1]["lam"])):
        sub = agg[(agg.lam == v["lam"]) & (agg.abar == v["abar"]) & (agg.method == "Tuned threshold + filter")]
        rows += (f"| {v['lam']:.3f} | {v['abar']} | {v['margin']:.4f} | {'yes' if v['ok'] else 'no'} | "
                 + " | ".join(f"{sub[sub.scenario == s].rmse.values[0]:.3f} / {100 * sub[sub.scenario == s].interv.values[0]:.1f} / "
                              + (f"{sub[sub.scenario == s].bound.values[0]:.2f}" if np.isfinite(sub[sub.scenario == s].bound.values[0]) else "—")
                              for s in ["E1 nominal", "E3 loss 0.5", "E5 bias+net", "E2 mult (HD)"]) + " |\n")
    T["TABLE_E13"] = (": TABLE XV. Safety–performance trade-off (E13, tuned threshold supervisor, 15 seeds). Each cell: "
                      "RMSE / interventions % / mean a-posteriori Theorem-2 bound on ‖c‖_P (second half of the episode; "
                      "compare ℓ = 2.79; — where alarms end the steps covered by Theorem 2 before the second half). Inequality (13), with η at buffered steps, is checked on every step before the first alarm "
                      "(the steps covered by Theorem 1 and Corollary 1); it held on all checked steps in "
                      + ", ".join(f"{k} {100 * v:.0f} %" for k, v in T["e13_thm2_by_scen"].items()) +
                      " of the trajectories (fraction of steps checked: "
                      + ", ".join(f"{k} {100 * v:.0f} %" for k, v in T["e13_thm2_checked"].items()) + ").\n\n| λ | ā | certificate margin (C2) | certified | "
                      "E1 | E3 loss 0.5 | E5 bias+net | E2 mult† |\n|---|---|---|---|---|---|---|---|\n" + rows)


# ------------------------------------------------------------------ chain
def chain(T):
    p = os.path.join(OUT, "chain_decentral.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    sims = pd.DataFrame(d["simulations"])
    rows = ""
    for c in d["certificates"]:
        s = sims[sims.N == c["N"]]
        rows += (f"| {c['N']} | {2 * c['N']} | {'hold' if c['all_local_ok'] else 'fail'} | {c['C2_margin']:.4f} | "
                 f"{c['coupling_bound'][1]:.4f} | "
                 + (f"{100 * s.viol_any.mean():.0f} | {100 * s.contain.min():.1f} | {100 * s.infeas.max():.2f} | "
                    f"{s.t_local_ms.mean():.2f} | {s.t_total_ms.mean():.1f} |" if len(s) else "— | — | — | — | — |") + "\n")
    T["TABLE_CHAIN"] = (": TABLE XVII. Compositional certificate and decentralised SAFE-Switch on chains of N coupled "
                        "motors (k_c = 0.05; simulations: 5 seeds × {network faults; burst bias + network faults} per N). "
                        "Time: mean per local controller, and summed over all controllers (one core; local controllers "
                        "are independent and parallelisable).\n\n| N | n | local certificates | min C2 margin | coupling "
                        "bound (speed) | viol. % | min contain. % | max infeas. % | ms per local step | ms per step (sum) |\n"
                        "|---|---|---|---|---|---|---|---|---|---|\n" + rows)
    T["chain"] = d["certificates"]
    T["chain_sims"] = dict(viol=float(sims.viol_any.mean()), n=int(len(sims)), contain_min=float(sims.contain.min()),
                           t_local=float(sims.t_local_ms.mean()))


if __name__ == "__main__":
    T = {}
    for f in (main_stats, adv, certs, e12, e15, tank, e13, chain):
        try:
            f(T)
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    json.dump(T, open(os.path.join(OUT, "tables_v2.json"), "w"), indent=1, default=float)
