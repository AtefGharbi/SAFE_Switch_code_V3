"""Builds all result tables (markdown) and headline numbers -> results/tables.json."""
import json, os, pickle
import numpy as np
import pandas as pd
from scipy import stats
from figs_results import load, ci, holm, METHODS, OUT

SETMETHODS = {"SAFE-Switch", "Risk-threshold + filter", "PPO w/o filter"}  # log containment


def fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def big(x):
    if not np.isfinite(x):
        return "—"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.3f}"


def main_tables(T):
    df, g = load("main")
    sel = ["E1 nominal", "E2 mult (HD)", "E2 bias 0.3", "E2 bias 2.0", "E2 ramp 0.2/s", "E2 stealth 0.03",
           "E3 loss 0.5", "E3 burst (uncapped)", "E3 actuator loss 0.3", "E4 delay 3", "E4 delay 10",
           "E5 bias+net", "E5 mult+net", "E7 unc 0.3"]
    out_of_env = {"E2 mult (HD)", "E3 burst (uncapped)", "E4 delay 10", "E5 mult+net", "E7 unc 0.3"}
    hdr = "| Scenario | " + " | ".join(METHODS) + " |\n|---|" + "---|" * len(METHODS) + "\n"
    rows = ""
    for s in sel:
        cells = []
        means = {m: g[(g.scenario == s) & (g.method == m)]["rmse"].mean() for m in METHODS}
        best = min(means.values())
        for m in METHODS:
            v = g[(g.scenario == s) & (g.method == m)]
            mu = v["rmse"].mean(); vi = 100 * v["viol_any"].mean()
            txt = f"{big(mu)} ({vi:.0f})"
            if abs(mu - best) < 1e-9:
                txt = f"**{txt}**"
            cells.append(txt)
        rows += f"| {s}{'†' if s in out_of_env else ''} | " + " | ".join(cells) + " |\n"
    T["TABLE_MAIN"] = (": TABLE VI. RMSE of ‖x‖ (mean over 20 paired seeds) and, in parentheses, the percentage of "
                       "episodes with a constraint violation. Best mean in bold. † = assumptions violated by design (Λ on both channels, "
                       "uncapped bursts, delay > 3, uncertainty > 3 %). The attack families without † use attack intervals ≥ 0.5 s, "
                       "for which A4 is not guaranteed, so their results are empirical (SAFE-Switch containment 77–90 %). "
                       "Only E1, E2-stealth, E3-loss, E3-actuator, and E4-delay ≤ 3 lie fully inside the envelope. SAFE-Switch and PPO entries are averaged over policy seeds (5 and 2).\n\n" + hdr + rows)
    # detailed E5 table
    s = "E5 bias+net"
    hdr = ("| Method | RMSE [95 % CI] | max ‖x‖ | viol. % | recovery (s) | energy | switches | interv. % | infeas. % | "
           "contain. % | ms/step |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = ""
    for m in METHODS:
        v = g[(g.scenario == s) & (g.method == m)]
        mu, lo, hi = ci(v["rmse"])
        cont = f"{100 * v['contain'].mean():.1f}" if m in SETMETHODS else "n/a"
        rows += (f"| {m} | {big(mu)} [{big(lo)}, {big(hi)}] | {big(v['max_dev'].mean())} | {100 * v['viol_any'].mean():.0f} | "
                 f"{fmt(np.nanmean(v['recovery']), 2)} | {big(v['energy'].mean())} | {v['switches'].mean():.1f} | "
                 f"{100 * v['interv'].mean():.1f} | {100 * v['infeas'].mean():.2f} | {cont} | {v['t_ms'].mean():.2f} |\n")
    T["TABLE_E5"] = (": TABLE VII. Detailed metrics for the combined scenario E5 (bias 1.0 on one channel, bursty loss 0.2, "
                     "delay ≤ 3, actuator loss 0.1); means over 20 paired seeds.\n\n" + hdr + rows)
    # win/tie/loss with Wilcoxon + Holm per comparison family
    scen = sorted(g.scenario.unique())
    hdr = "| Comparison (SAFE-Switch vs.) | better | no significant difference | worse | median paired ΔRMSE |\n|---|---|---|---|---|\n"
    rows = ""
    wtl = {}
    for m in METHODS[1:]:
        ps, ds = [], []
        for s_ in scen:
            a = g[(g.scenario == s_) & (g.method == "SAFE-Switch")].sort_values("seed")["rmse"].values
            b = g[(g.scenario == s_) & (g.method == m)].sort_values("seed")["rmse"].values
            d = a - b
            ds.append(np.median(d))
            try:
                p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue if np.any(np.abs(d) > 1e-12) else 1.0
            except ValueError:
                p = 1.0
            ps.append(p)
        pa = holm(np.array(ps))
        w = sum((pa < 0.05) & (np.array(ds) < 0)); l = sum((pa < 0.05) & (np.array(ds) > 0))
        wtl[m] = (int(w), int(len(scen) - w - l), int(l))
        rows += f"| {m} | {w} | {len(scen) - w - l} | {l} | {np.median(ds):+.4f} |\n"
    T["TABLE_WTL"] = (f": TABLE VIII. Paired comparison over all {len(scen)} scenario families (Wilcoxon signed-rank on "
                      "per-seed RMSE, Holm-corrected within each row, α = 0.05).\n\n" + hdr + rows)
    T["wtl"] = wtl
    # headline numbers
    h = {}
    for m in METHODS:
        v = g[g.method == m]
        h[m] = dict(viol_eps=int(round((v["viol_any"]).sum())), n_eps=int(len(v)),
                    div_eps=int(round(v["diverged"].sum())))
    inenv = ["E1 nominal", "E2 stealth 0.03", "E3 loss 0.1", "E3 loss 0.3", "E3 loss 0.5", "E3 actuator loss 0.3",
             "E4 delay 1", "E4 delay 3"]
    for m in METHODS:
        v = g[(g.method == m) & (g.scenario.isin(inenv))]
        h[m]["viol_eps_inenv"] = int(round(v["viol_any"].sum())); h[m]["n_inenv"] = int(len(v))
    T["headline"] = h
    T["inenv"] = inenv
    e5 = {m: g[(g.scenario == "E5 bias+net") & (g.method == m)]["rmse"].mean() for m in METHODS}
    T["e5"] = e5
    T["n_scen"] = len(scen)
    # per-policy-seed spread for SAFE-Switch on E5
    d5 = df[(df.scenario == "E5 bias+net") & (df.method == "SAFE-Switch")].groupby("policy_seed")["rmse"].mean()
    T["e5_seed_spread"] = [float(x) for x in d5.values]
    # containment within envelope for SAFE
    T["contain"] = {s_: float(g[(g.scenario == s_) & (g.method == "SAFE-Switch")]["contain"].mean()) for s_ in scen}
    T["infeas"] = {s_: float(g[(g.scenario == s_) & (g.method == "SAFE-Switch")]["infeas"].mean()) for s_ in scen}
    T["tms"] = {m: float(g[g.method == m]["t_ms"].mean()) for m in METHODS}
    T["tmax"] = {m: float(df[df.method == m]["t_max_ms"].max()) for m in METHODS}


def ablation_table(T):
    df, g = load("ablation")
    _, gm = load("main")
    from experiments import ABL_SCEN
    full = gm[(gm.method == "SAFE-Switch") & (gm.scenario.isin(ABL_SCEN))]
    g = pd.concat([g, full])
    variants = ["SAFE-Switch", "w/o risk features", "w/o timestamps", "w/o dwell time", "w/o packetized fallback",
                "w/o zonotope (box only)", "w/o quarantine/re-anchoring", "Filter only (N)", "Robust mode + filter (R)",
                "Random supervisor + filter"]
    gp = gm[(gm.method == "PPO w/o filter") & (gm.scenario.isin(ABL_SCEN))].copy()
    gp["method"] = "w/o safety filter"
    g = pd.concat([g, gp])
    variants.insert(1, "w/o safety filter")
    hdr = "| Variant | " + " | ".join(ABL_SCEN) + " | switches (E5) | contain. % (E5) |\n|---|" + "---|" * (len(ABL_SCEN) + 2) + "\n"
    rows = ""
    for v in variants:
        cells = []
        for s in ABL_SCEN:
            x = g[(g.scenario == s) & (g.method == v)]
            cells.append(f"{big(x['rmse'].mean())} ({100 * x['viol_any'].mean():.0f})" if len(x) else "—")
        e5 = g[(g.scenario == "E5 bias+net") & (g.method == v)]
        rows += f"| {v} | " + " | ".join(cells) + f" | {e5['switches'].mean():.1f} | {100 * e5['contain'].mean():.1f} |\n"
    T["TABLE_ABL"] = (": TABLE IX. Ablation (E6): RMSE of ‖x‖ (violation-episode %) with one component removed. "
                      "PPO-based variants use policy seeds 0–2 (risk features: 0–1; safety filter: the two policies "
                      "trained without filter).\n\n" + hdr + rows)
    T["abl"] = {v: {s: float(g[(g.scenario == s) & (g.method == v)]["rmse"].mean()) for s in ABL_SCEN} for v in variants}
    T["abl_contain"] = {v: float(g[(g.scenario == "E5 bias+net") & (g.method == v)]["contain"].mean()) for v in variants}


def stress_table(T):
    if not os.path.exists(os.path.join(OUT, "stress.jsonl")):
        return
    df, g = load("stress")
    meths = ["SAFE-Switch", "PPO w/o filter", "Filter only (N)", "Always N w/o filter", "Random supervisor + filter",
             "Random supervisor w/o filter", "Risk-threshold + filter", "Huang–Dong [1]", "LQR", "Robust H∞", "Robust MPC"]
    sc = list(g.scenario.unique())
    sc.sort()
    hdr = "| Method | " + " | ".join(sc) + " |\n|---|" + "---|" * len(sc) + "\n"
    rows = ""
    for m in meths:
        cells = []
        for s in sc:
            x = g[(g.scenario == s) & (g.method == m)]
            cells.append(f"{big(x['rmse'].mean())} ({100 * x['viol_any'].mean():.0f})" if len(x) else "—")
        rows += f"| {m} | " + " | ".join(cells) + " |\n"
    T["TABLE_STRESS"] = (": TABLE X. Worst-case-supervisor stress test (E10): bias of magnitude 6 in short bursts "
                         "(mean 0.1 s, every ≈1 s) and, †, in long intervals (outside A4). Entries: RMSE of ‖x‖ "
                         "(violation-episode %), 20 paired seeds.\n\n" + hdr + rows)
    T["stress"] = {m: {s: [float(g[(g.scenario == s) & (g.method == m)]["rmse"].mean()),
                           float(g[(g.scenario == s) & (g.method == m)]["viol_any"].mean()),
                           float(g[(g.scenario == s) & (g.method == m)]["infeas"].mean())] for s in sc} for m in meths}


def lam_table(T):
    if not os.path.exists(os.path.join(OUT, "lam_sweep.json")):     # V1 study (E9), superseded by E13
        return
    d = pd.DataFrame(json.load(open(os.path.join(OUT, "lam_sweep.json"))))
    t = d.groupby(["scenario", "sup", "lam"]).agg(rmse=("rmse", "mean"), interv=("interv", "mean"),
                                                   g95=("gamma_p95", "mean"), bound=("bound", "mean"),
                                                   viol=("viol_any", "mean")).reset_index()
    hdr = ("| Scenario | supervisor | λ | RMSE | interventions % | p95 of γ_k | Theorem-2 bound on ‖c‖_P | viol. % |\n"
           "|---|---|---|---|---|---|---|---|\n")
    rows = ""
    for _, r in t.iterrows():
        sup = {"always_N": "always N", "threshold": "risk threshold"}[r.sup]
        rows += (f"| {r.scenario} | {sup} | {r.lam:.3f} | {r.rmse:.3f} | {100 * r.interv:.1f} | {r.g95:.4f} | "
                 f"{r.bound:.2f} | {100 * r.viol:.0f} |\n")
    T["TABLE_LAM"] = (": TABLE XI. Contraction-rate study (E9, 15 seeds). The bound is (ε₀ + p95(γ))/(1 − λ), to be "
                      "compared with ℓ = 2.79.\n\n" + hdr + rows)
    T["lam"] = t.to_dict(orient="records")


def scal_table(T):
    p = os.path.join(OUT, "scal.json")
    if not os.path.exists(p):
        return
    r = json.load(open(p))
    hdr = ("| N motors | n | m | set estimator (ms) | Kalman (ms) | safety filter (ms) | PPO (ms) | total mean (ms) | "
           "synthesis (s) | certificate (C1)–(C5) |\n|---|---|---|---|---|---|---|---|---|---|\n")
    rows = ""
    for x in r:
        rows += (f"| {x['N']} | {x['n']} | {x['m']} | {x['est_ms'][0]:.2f} (p99 {x['est_ms'][1]:.2f}) | "
                 f"{x['kf_ms'][0]:.2f} | {x['filter_ms'][0]:.2f} (p99 {x['filter_ms'][1]:.2f}) | {x['ppo_ms'][0]:.3f} | "
                 f"{x['total_ms']:.2f} | {x['synthesis_s']:.2f} | {'holds' if x['cert_ok'] else 'fails (α = %.3f)' % x['alpha']} |\n")
    T["TABLE_SCAL"] = (": TABLE XII. Scalability (E8) on chains of coupled DC motors; per-step time on one CPU core "
                       "(Python/NumPy; filter: exact 1-D projection for m = 1, Clarabel SOCP for m > 1).\n\n" + hdr + rows)
    T["scal"] = r


def risk_table(T):
    r = json.load(open(os.path.join(OUT, "risk_model.json")))
    T["risk"] = {k: r[k] for k in ("auc", "brier", "ece", "tpr", "fpr", "base_rate", "n_train", "n_test")}
    T["TABLE_RISK"] = (": TABLE II. Attack-risk model on held-out episodes (step-level labels: attack interval active).\n\n"
                       "| Training steps | test steps | attack base rate | ROC AUC | Brier score | ECE (10 bins) | TPR @ 0.5 | FPR @ 0.5 |\n"
                       "|---|---|---|---|---|---|---|---|\n"
                       f"| {r['n_train']} | {r['n_test']} | {r['base_rate']:.3f} | {r['auc']:.3f} | {r['brier']:.3f} | "
                       f"{r['ece']:.3f} | {r['tpr']:.3f} | {r['fpr']:.3f} |\n")


def repro_table(T):
    r = json.load(open(os.path.join(OUT, "repro_hd.json")))
    gg = r["gains"]
    T["repro"] = r
    T["TABLE_REPRO"] = (": TABLE III-A. Reproduction of the DC-motor design of [1].\n\n"
                        "| Quantity | reported in [1] (learned) | this reproduction (model-based) |\n|---|---|---|\n"
                        f"| mode-0 gain $K_0$ ($u=-K_0x$) | [{gg['K0_paper'][0]:.4f}, {gg['K0_paper'][1]:.4f}] | [{gg['K0'][0]:.4f}, {gg['K0'][1]:.4f}] |\n"
                        f"| mode-1 gain $K_1$ ($u=-K_1\\tilde x$) | [{gg['K1_paper'][0]:.4f}, {gg['K1_paper'][1]:.4f}] | [{gg['K1'][0]:.4f}, {gg['K1'][1]:.4f}] |\n"
                        f"| $P_0$ (entries 11, 12, 22) | 0.1557, 0.1494, 0.2776 | {gg['P0'][0][0]:.4f}, {gg['P0'][0][1]:.4f}, {gg['P0'][1][1]:.4f} |\n"
                        f"| non-switched controller under fault | diverges (Fig. 9 of [1]) | diverges (‖x‖ > 10³ at t = {r['nonswitched']['t_exceed_1e3']:.1f} s) |\n"
                        f"| switched controller, max ‖x‖ over 800 s | bounded (Fig. 7 of [1]) | {r['switched']['max_norm']:.3f} ({r['switched']['n_switch']} switches) |\n"
                        f"| effect of η∞ (50 → 0.05) on max ‖x‖ | chattering reduced (Fig. 8 of [1]) | "
                        f"{r['eta_sweep']['50']['max_norm']:.2f} → {r['eta_sweep']['0.05']['max_norm']:.2f} |\n")


def cert_table(T):
    from config import Design
    D = Design(); s = D.summary(); C = D.cert
    lhs = C.alpha * D.ell + max(C.beta - C.alpha, 0) * C.qbar + C.c0
    T["cert"] = dict(s, lhs=lhs)
    T["TABLE_CERT"] = (": TABLE III. Certificate quantities for the DC motor (Remark 3).\n\n"
                       "| Quantity | value | Quantity | value |\n|---|---|---|---|\n"
                       f"| $K_R$ | [{s['KR'][0]:.4f}, {s['KR'][1]:.4f}] | $\\rho$ | {s['rho']:.4f} |\n"
                       f"| $P$ | [[{s['P'][0][0]:.4f}, {s['P'][0][1]:.4f}], [{s['P'][1][0]:.4f}, {s['P'][1][1]:.4f}]] | $\\alpha$ (C1) | {s['alpha']:.4f} |\n"
                       f"| $\\ell$ | {s['ell']:.4f} | $\\beta$ | {s['beta']:.4f} |\n"
                       f"| $\\sigma\\ell$ (bound on $|x|$ in Ω) | [{s['sigma'][0] * s['ell']:.3f}, {s['sigma'][1] * s['ell']:.3f}] | $c_0$ | {s['c0']:.4f} |\n"
                       f"| $\\kappa\\ell$ (bound on $|K_Rx|$) | {s['kappa'] * s['ell']:.3f} | $\\bar r$ | [{s['rbar'][0]:.4f}, {s['rbar'][1]:.4f}] |\n"
                       f"| $L=N_b+d_{{\\max}}+N_a$ | {s['L']} | $\\bar q=p^\\top\\bar r$ (C3: $<\\ell$) | {s['qbar']:.4f} |\n"
                       f"| $\\lambda$, $\\epsilon_0$ (C5: $\\lambda\\ge\\rho$) | {s['lam']}, {s['eps0']} | (C2) LHS vs. $\\ell$ | {lhs:.4f} ≤ {s['ell']:.4f} |\n")


if __name__ == "__main__":
    T = {}
    for f in (risk_table, repro_table, cert_table, main_tables, ablation_table, stress_table, lam_table, scal_table):
        try:
            f(T)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    json.dump(T, open(os.path.join(OUT, "tables.json"), "w"), indent=1, default=float)
    for k, v in T.items():
        if k.startswith("TABLE"):
            print(v)
    print(json.dumps({k: T[k] for k in ("headline", "e5", "wtl", "e5_seed_spread")}, indent=1, default=float))
