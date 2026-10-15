"""Plant models used in the SAFE-Switch study.

Benchmark 1: DC motor speed-control error dynamics of Huang & Dong (2020), Table II.
Benchmark 2: single-area load-frequency control (LFC) plant (generalization study).

All plants are written as continuous-time (A, B, D) and discretized with a
zero-order hold. Parametric uncertainty is represented by element-wise bounds
|A_true - A_hat| <= dA, |B_true - B_hat| <= dB obtained by vertex enumeration of
the physical-parameter box.
"""
import itertools
import numpy as np
from scipy.linalg import expm


def c2d(A, B, D, Ts):
    n, m, p = A.shape[0], B.shape[1], D.shape[1]
    M = np.zeros((n + m + p, n + m + p))
    M[:n, :n] = A
    M[:n, n:n + m] = B
    M[:n, n + m:] = D
    E = expm(M * Ts)
    return E[:n, :n], E[:n, n:n + m], E[:n, n + m:]


# ---------------------------------------------------------------- DC motor
DCM_NOMINAL = dict(CT=0.06, ML=0.1, LT=0.3, RT=2.07, J=80.45e-4, CV=6.27e-3)
XI = 2.0039  # input scaling of Huang & Dong


def dcmotor_ct(CT=0.06, LT=0.3, RT=2.07, J=80.45e-4, CV=6.27e-3, **_):
    """Error dynamics x=[dI, dOmega]; u=(U - CV*Od - RT*Id)/xi (Huang & Dong, Sec. IV)."""
    A = np.array([[-RT / LT, -CV / LT], [CT / J, 0.0]])
    B = np.array([[XI / LT], [0.0]])
    D = np.array([[0.1], [0.3]])  # B_d of Huang & Dong
    return A, B, D


class Plant:
    """Discrete-time uncertain LTI plant x+ = A x + B u + E w."""

    def __init__(self, name, A_ct_fun, nominal, Ts, rel_unc, unc_keys):
        self.name = name
        self.Ts = Ts
        self.fun = A_ct_fun
        self.nominal = dict(nominal)
        self.Ac, self.Bc, self.Dc = A_ct_fun(**nominal)
        self.A, self.B, self.E = c2d(self.Ac, self.Bc, self.Dc, Ts)
        self.n, self.m = self.B.shape
        self.rel_unc = rel_unc
        self.unc_keys = unc_keys
        self.dA, self.dB = self.uncertainty_bounds(rel_unc)

    def matrices(self, params):
        Ac, Bc, Dc = self.fun(**params)
        return c2d(Ac, Bc, Dc, self.Ts)

    def uncertainty_bounds(self, rel):
        dA = np.zeros_like(self.A)
        dB = np.zeros_like(self.B)
        dE = np.zeros_like(self.E)
        self.dE = dE
        if rel <= 0:
            return dA, dB
        # vertex enumeration plus interior samples of the parameter box
        grids = [np.linspace(1 - rel, 1 + rel, 5) for _ in self.unc_keys]
        for scal in itertools.product(*grids):
            p = dict(self.nominal)
            for k, s in zip(self.unc_keys, scal):
                p[k] = self.nominal[k] * s
            A, B, E = self.matrices(p)
            dA = np.maximum(dA, np.abs(A - self.A))
            dB = np.maximum(dB, np.abs(B - self.B))
            dE = np.maximum(dE, np.abs(E - self.E))
        self.dE = dE * 1.02  # V3: disturbance-matrix uncertainty |E_true - E_hat| <= dE
        return dA * 1.02, dB * 1.02  # 2 % margin for gridding

    def sample_true(self, rng, rel):
        p = dict(self.nominal)
        for k in self.unc_keys:
            p[k] = self.nominal[k] * (1 + rng.uniform(-rel, rel))
        A, B, E = self.matrices(p)
        return A, B, E, p


def make_dcmotor(Ts=0.05, rel_unc=0.05):
    return Plant("dcmotor", dcmotor_ct, DCM_NOMINAL, Ts, rel_unc, ["RT", "CT", "J", "LT"])


# ---------------------------------------------------------------- LFC (benchmark 2)
LFC_NOMINAL = dict(Tg=0.08, Tt=0.30, Tp=20.0, Kp=120.0, Rd=2.4)


def lfc_ct(Tg=0.08, Tt=0.30, Tp=20.0, Kp=120.0, Rd=2.4, **_):
    """Single-area LFC: x=[df, dPg, dXg, ACE-integral]; u = governor set-point; w = load.
    Standard non-reheat thermal area (Bevrani, Robust Power System Frequency Control)."""
    A = np.array([
        [-1 / Tp, Kp / Tp, 0.0, 0.0],
        [0.0, -1 / Tt, 1 / Tt, 0.0],
        [-1 / (Rd * Tg), 0.0, -1 / Tg, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    B = np.array([[0.0], [0.0], [1 / Tg], [0.0]])
    D = np.array([[-Kp / Tp], [0.0], [0.0], [0.0]])
    return A, B, D


def make_lfc(Ts=0.05, rel_unc=0.05):
    return Plant("lfc", lfc_ct, LFC_NOMINAL, Ts, rel_unc, ["Tg", "Tt", "Tp", "Kp"])


# ---------------------------------------------------------------- coupled tanks (benchmark 2)
TANK_NOMINAL = dict(A1=15.5, A2=15.5, a1=0.178, a2=0.178, Kp=3.3, h1o=15.0, h2o=15.0, g=981.0)


def tank_ct(A1=15.5, A2=15.5, a1=0.178, a2=0.178, Kp=3.3, h1o=15.0, h2o=15.0, g=981.0, **_):
    """Cascaded two-tank process (Quanser-type coupled tanks) linearised at (h1o, h2o):
    x = [dh1, dh2] (cm), u = pump-voltage deviation (V), w = inflow disturbance (cm^3/s)."""
    k1 = a1 / A1 * np.sqrt(g / (2 * h1o))
    k12 = a1 / A2 * np.sqrt(g / (2 * h1o))
    k2 = a2 / A2 * np.sqrt(g / (2 * h2o))
    A = np.array([[-k1, 0.0], [k12, -k2]])
    B = np.array([[Kp / A1], [0.0]])
    D = np.array([[1.0 / A1], [0.0]])
    return A, B, D


def make_tank(Ts=0.5, rel_unc=0.03):
    return Plant("tank", tank_ct, TANK_NOMINAL, Ts, rel_unc, ["A1", "A2", "a1", "a2", "Kp"])
