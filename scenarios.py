"""Threat and network scenario generator (sensor FDI, packet loss, delay, actuator loss).

A scenario is fully pre-sampled from a seed so that every controller is evaluated on
exactly the same disturbance, noise, attack, loss and delay realisation (paired design).
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ScenarioCfg:
    name: str = "S0"
    T: int = 600                     # steps (Ts = 0.05 s -> 30 s)
    attack: str = "none"             # none | mult | bias | ramp | stealth
    attack_mag: float = 0.0          # bias magnitude / ramp slope (per s) / stealth amplitude
    Lam: tuple = (-1.5, -0.5)        # multiplicative gains (Huang & Dong)
    attack_channels: tuple = (0, 1)
    on_mean: float = 4.0             # mean attack-on duration [s]
    off_mean: float = 6.0            # mean attack-off duration [s]
    first_attack: float = 5.0        # earliest attack onset [s]
    min_on: float = 0.5              # minimum attack-on duration [s]
    p_loss: float = 0.0              # sensor loss prob (good state of Gilbert-Elliott)
    burst: bool = False              # Gilbert-Elliott bursts
    p_gb: float = 0.05
    p_bg: float = 0.3
    Nb_cap: int = 4                  # cap on consecutive sensor losses (None = uncapped)
    dmax: int = 0                    # max sensor delay (steps)
    pa_loss: float = 0.0             # actuator loss probability
    Na_cap: int = 3
    rel_unc: float = 0.03            # true-parameter perturbation (fraction)
    x0: tuple = (1.0, 1.0)
    x0_random: bool = False
    w_sigma: float = 0.3
    load_steps: bool = True
    wbar: float = 1.0
    vbar: float = 0.02
    Ts: float = 0.05                 # sampling period of the benchmark
    time_scale: float = 1.0          # multiplies on/off/first/min_on durations
    w_scale: float = 1.0             # multiplies the disturbance (and its bound)


def _intervals(rng, T, Ts, on_mean, off_mean, first, min_on=0.5):
    act = np.zeros(T, bool)
    t = first + rng.exponential(off_mean * 0.3)
    while t < T * Ts:
        d_on = max(min_on, rng.exponential(on_mean))
        k0, k1 = int(t / Ts), int(min(T, (t + d_on) / Ts))
        act[k0:k1] = True
        t += d_on + max(0.5, rng.exponential(off_mean))
    return act


def _losses(rng, T, p, burst, p_gb, p_bg, cap):
    lost = np.zeros(T, bool)
    bad = False
    run = 0
    for k in range(T):
        if burst:
            bad = (rng.random() < p_gb) if not bad else (rng.random() >= p_bg)
        l = bad or (rng.random() < p)
        if cap is not None and l and run >= cap:
            l = False
        lost[k] = l
        run = run + 1 if l else 0
    return lost


def sample(cfg: ScenarioCfg, seed: int, n: int = 2, Ts: float = None, xmax=None):
    Ts = cfg.Ts if Ts is None else Ts
    rng = np.random.default_rng(seed)
    T = cfg.T
    sc = {"cfg": cfg, "seed": seed, "T": T}
    # attack schedule
    if cfg.attack == "none":
        act = np.zeros(T, bool)
    else:
        ts_ = cfg.time_scale
        act = _intervals(rng, T, Ts, cfg.on_mean * ts_, cfg.off_mean * ts_, cfg.first_attack * ts_,
                         cfg.min_on * ts_)
    sc["attack_on"] = act
    chans = np.zeros(n, bool)
    chans[list(cfg.attack_channels)] = True
    sc["attack_chans"] = chans
    sign = rng.choice([-1.0, 1.0], size=n)
    sc["attack_sign"] = sign
    sc["stealth_phase"] = rng.uniform(0, 2 * np.pi, size=n)
    # sensor losses / delays per channel
    sc["s_lost"] = np.stack([_losses(rng, T, cfg.p_loss, cfg.burst, cfg.p_gb, cfg.p_bg, cfg.Nb_cap)
                             for _ in range(n)], 1)
    sc["s_delay"] = rng.integers(0, cfg.dmax + 1, size=(T, n)) if cfg.dmax > 0 else np.zeros((T, n), int)
    sc["a_lost"] = _losses(rng, T, cfg.pa_loss, False, 0, 1, cfg.Na_cap)
    sc["a_lost"][0] = False
    # disturbance: bounded noise + piecewise-constant load steps, |w| <= wbar
    w = np.clip(rng.normal(0, cfg.w_sigma, T), -cfg.wbar, cfg.wbar)
    if cfg.load_steps:
        lvl = 0.0
        load = np.zeros(T)
        for k in range(T):
            if rng.random() < 1 / 120:
                lvl = rng.choice([-0.5, 0.0, 0.5])
            load[k] = lvl
        w = np.clip(w + load, -cfg.wbar, cfg.wbar)
    sc["w"] = (cfg.w_scale * w)[:, None]
    sc["Ts"] = Ts
    sc["v"] = rng.uniform(-cfg.vbar, cfg.vbar, size=(T, n))
    sc["param_scale"] = rng.uniform(-1, 1, size=8)
    if cfg.x0_random:
        sc["x0"] = rng.uniform(-1, 1, n) * np.array(xmax if xmax is not None else np.ones(n)) * 0.45
    else:
        sc["x0"] = np.array(cfg.x0, float)
    return sc


def corrupt(sc, k, x, v):
    """Sensor-side measurement generated at time k (before the network)."""
    cfg = sc["cfg"]
    y = x + v
    M = sc.get("auth_period")
    if not sc["attack_on"][k] or (M and k % M == 0):  # authenticated packets cannot be modified
        return y, np.zeros_like(y)
    ch = sc["attack_chans"]
    a = np.zeros_like(y)
    if cfg.attack == "mult":
        Lam = np.array(cfg.Lam)
        a = (Lam - 1.0) * y
    elif cfg.attack == "bias":
        a = sc["attack_sign"] * cfg.attack_mag
    elif cfg.attack == "ramp":
        # slope per second, measured from onset of the current attack interval
        on = sc["attack_on"][:k + 1][::-1]
        dur = np.argmin(on) if not on.all() else len(on)
        a = sc["attack_sign"] * cfg.attack_mag * dur * sc.get("Ts", 0.05)
    elif cfg.attack == "gate":
        # white-box adaptive attack: keep y at 95 % of the acceptance gate of the set estimator,
        # on the side that pushes the estimate away from the true state
        ctrl = sc.get("_ctrl")
        if ctrl is not None and hasattr(ctrl, "est") and (k - 1) in ctrl.est.c:
            est = ctrl.est
            cp, rp = est.C.predict(est.c[k - 1], est.r[k - 1], np.atleast_1d(sc.get("_u_prev", 0.0)))
            target = cp + sc["attack_sign"] * 0.95 * (rp + est.delta)
            a = np.clip(target - y, -cfg.attack_mag, cfg.attack_mag)
        else:  # no consistency gate to ride: inject the full admissible bias
            a = sc["attack_sign"] * cfg.attack_mag
    elif cfg.attack == "stealth":
        a = cfg.attack_mag * np.sin(0.5 * k * sc.get("Ts", 0.05) / cfg.time_scale + sc["stealth_phase"])
    a = np.where(ch, a, 0.0)
    return y + a, a
