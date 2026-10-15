"""Benchmark-specific Design factories and scenario profiles."""
import numpy as np
from config import Design

PROFILES = {
    "dcmotor": dict(T=600, Ts=0.05, time_scale=1.0, w_scale=1.0),
    "tank": dict(T=240, Ts=0.5, time_scale=5.0, w_scale=2.0),
}


def make_design(bench="dcmotor", **kw):
    if bench == "dcmotor":
        return Design(**kw)
    if bench == "tank":
        base = dict(bench="tank", Ts=0.5, rel_unc=0.03, Nb=4, dmax=3, Na=3, vbar=0.02, abar=0.03, wbar=2.0,
                    umax=5.0, xmax=(6.0, 6.0), QN=np.diag([1.0, 100.0]), RN=1.0, QR=np.diag([1.0, 3.0]),
                    RR=10.0, QM=np.diag([1.0, 30.0]), eps0=0.02)
        base.update(kw)
        return Design(**base)
    raise ValueError(bench)
