"""Figures for the manuscript (static PNG, 300 dpi, print/light)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "results")
FIG = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIG, exist_ok=True)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = {  # entity -> fixed categorical slot (reference palette order)
    "SAFE-Switch": "#2a78d6", "Huang–Dong [1]": "#eb6834", "LQR": "#1baf7a", "Robust H∞": "#eda100",
    "Robust MPC": "#e87ba4", "Risk-threshold + filter": "#008300", "PPO w/o filter": "#6250d6",
}
ATTACK_SHADE = "#f3d6d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.4,
    "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlesize": 8.5, "axes.titleweight": "bold",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


# ------------------------------------------------------------------ architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 57); ax.axis("off")

    def box(x, y, w, h, txt, fc="#f4f6fb", ec="#2a78d6", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.3, color=INK, linespacing=1.35,
                fontweight="bold" if bold else "normal", wrap=True)

    def arr(x0, y0, x1, y1, txt=None, dy=1.2, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9,
                                     color=INK2, linestyle=ls))
        if txt:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, ha="center", va="bottom", fontsize=6, color=INK2)

    # plant and network
    box(1, 44, 16, 9, "DC motor plant\n(uncertain $A,B$;\nload $w_k$)", fc="#ffffff", ec=MUTED)
    box(21, 44, 20, 9, "Sensor network\nFDI, loss, delay\npackets $(i,\\tau,y)$", fc="#fbf1f1", ec="#e34948")
    box(1, 17, 16, 9, "Actuator +\npacket buffer", fc="#ffffff", ec=MUTED)
    box(21, 17, 20, 9, "Actuator network\nloss + ACK", fc="#fbf1f1", ec="#e34948")
    box(46, 44, 24, 9, "Set estimator\n(zonotope/box, timestamps,\ngating, quarantine)")
    box(46, 31, 24, 8, "Kalman filter (OOSM)\nestimate $\\hat x_k$")
    box(76, 44, 23, 9, "Calibrated risk\n$\\rho_k=\\sigma(\\theta^\\top\\phi_k)$")
    box(76, 31, 23, 8, "PPO supervisor\n$s_k\\in\\{N,R\\}$ + dwell time", fc="#eef5ff")
    box(46, 17, 24, 9, "Candidate action\nN: $K_N\\hat x_k$,  R: $K_M c_k$")
    box(76, 15, 23, 11, "Robust safety filter\n(certified level $\\ell$)\ninfeasible $\\Rightarrow K_R c_k$", fc="#eef8f1", ec="#008300")
    box(46, 2, 24, 8, "Packet: $u^{safe}_k$ +\n$N_a$ fallback inputs")
    arr(17, 48.5, 21, 48.5)
    arr(41, 49, 46, 49)
    arr(41, 46, 46, 35)
    arr(70, 48.5, 76, 48.5)
    arr(70, 35, 76, 45.5)
    arr(87.5, 44, 87.5, 39)
    arr(76, 33, 70, 24.5)
    arr(58, 31, 58, 26)
    arr(71, 44.5, 78.5, 26.5, ls="--")
    arr(70, 21.5, 76, 21.5)
    arr(87.5, 15, 70, 6)
    arr(46, 6, 31, 17)
    arr(21, 21.5, 17, 21.5)
    arr(9, 26, 9, 44)
    ax.text(58, 55.2, "SAFE-Switch control station", fontsize=7.5, color=INK2, fontweight="bold", ha="center")
    save(fig, "fig1_architecture.png")


# ------------------------------------------------------------------ HD reproduction
def fig_repro():
    d = np.load(os.path.join(OUT, "repro_hd_traces.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    t = d["sw_t"]
    ax = axs[0]
    ax.fill_between(t, 0, d["sw_sig"], step="post", color=ATTACK_SHADE, label="fault active $\\sigma$")
    ax.step(t, d["sw_sigp"] * 0.8 + 0.1, where="post", color=COL["Huang–Dong [1]"], lw=1.0,
            label="controller mode $\\sigma'$ (scaled)")
    ax.set_xlabel("time (s)"); ax.set_yticks([0, 1]); ax.set_title("(a) fault and switching signals")
    ax.legend(loc="upper right", fontsize=6.5, ncol=2, bbox_to_anchor=(1, 1.22))
    ax = axs[1]
    ax.semilogy(t, d["sw_nx"], color=COL["Huang–Dong [1]"], label="switched H∞ (PPF law)")
    nx = d["ns_nx"]
    ax.semilogy(d["ns_t"], nx, color=COL["LQR"], label="non-switched H∞ (mode 0)")
    ax.set_ylim(1e-4, 1e12); ax.set_xlabel("time (s)"); ax.set_ylabel("$\\|x\\|$")
    ax.set_title("(b) state norm"); ax.legend(fontsize=6.5, loc="upper right")
    save(fig, "fig3_repro_hd.png")


def fig_risk():
    r = json.load(open(os.path.join(OUT, "risk_model.json")))
    rel = np.array(r["reliability"])
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.4))
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax.plot(rel[:, 0], rel[:, 1], "o-", color=COL["SAFE-Switch"], ms=4)
    ax.set_xlabel("predicted risk"); ax.set_ylabel("observed attack frequency")
    ax.set_title(f"(a) reliability (ECE = {r['ece']:.3f})")
    ax = axs[1]
    names = ["bias", "residual EWMA", "rejection EWMA", "KF–set gap", "KF outside set", "info age",
             "loss EWMA", "radius", "CUSUM"]
    th = np.array(r["theta"])
    y = np.arange(len(th))[::-1]
    ax.barh(y, th, color=[COL["SAFE-Switch"] if v >= 0 else COL["Huang–Dong [1]"] for v in th], height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5); ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title("(b) standardised coefficients"); ax.grid(axis="y", visible=False)
    save(fig, "fig_risk.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["arch", "repro", "risk"]
    if "arch" in which: fig_architecture()
    if "repro" in which: fig_repro()
    if "risk" in which: fig_risk()
