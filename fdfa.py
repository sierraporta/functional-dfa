"""
fdfa.py — Functional Detrended Fluctuation Analysis (FDFA)
===========================================================

Generalization of (MF)DFA in which the local fluctuation measure is not
hard-wired to the RMS/variance but is a pluggable dispersion functional
Phi acting on the detrending residuals of each window:

    F_Phi(v, s) = Phi( r_1, ..., r_s ),      r_i = Y(i) - P_m(i).

Classical DFA/MFDFA is recovered with Phi = RMS.

All built-in functionals are normalized to be *consistent for the standard
deviation under Gaussian residuals*, so that on Gaussian data every Phi
estimates the same local scale (identical log-log intercepts up to noise).
Normalization constants do not affect scaling exponents.

Admissibility (see the companion theory note): a functional is admissible if
it is positively homogeneous of degree 1, translation-insensitive on
detrended residuals, symmetric, non-degenerate, and concentrates over the
window ensemble. All functionals exported here are admissible; they differ
in moment requirements and breakdown points.

Author: David Sierra Porta (framework), code drafted with Claude.
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import numpy as np

__all__ = [
    "Functional",
    "rms", "make_lp", "l1", "mad", "make_quantile", "iqr",
    "make_huber", "linf", "make_trimmed_std", "entropy_power",
    "FUNCTIONALS",
    "fdfa", "hurst", "mfdfa_hq", "fdfa_multi", "mfdfa_hq_multi",
    "FDFAResult", "HqResult",
    "default_scales",
]

# ----------------------------------------------------------------------
# Gaussian consistency constants
# ----------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)


def _phi_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _norm_ppf(p: float, tol: float = 1e-12) -> float:
    """Inverse standard normal CDF (bisection; no scipy dependency)."""
    lo, hi = -10.0, 10.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _lp_gauss_const(p: float) -> float:
    """(E|Z|^p)^(1/p) for Z ~ N(0,1)."""
    return (2.0 ** (p / 2.0) * math.gamma((p + 1.0) / 2.0) / math.sqrt(math.pi)) ** (1.0 / p)


_MAD_CONST = 1.0 / _norm_ppf(0.75)          # 1.4826...
_ENTROPY_GAUSS = math.sqrt(2.0 * math.pi * math.e)  # exp(h(Z)) for N(0,1)


# ----------------------------------------------------------------------
# Functional container
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Functional:
    """A local dispersion functional Phi.

    fn maps a 2-D residual array of shape (n_windows, s) to a 1-D array
    of per-window fluctuation values (n_windows,). Must be positively
    homogeneous of degree 1 for scaling laws to hold.
    """
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    moment_requirement: float = 2.0   # minimal finite absolute moment needed
    breakdown: float = 0.0            # asymptotic breakdown point

    def __call__(self, R: np.ndarray) -> np.ndarray:
        R = np.atleast_2d(np.asarray(R, dtype=float))
        return self.fn(R)


# ----------------------------------------------------------------------
# Built-in functionals  (all sigma-consistent under Gaussian residuals)
# ----------------------------------------------------------------------

rms = Functional(
    "rms",
    lambda R: np.sqrt(np.mean(R ** 2, axis=1)),
    moment_requirement=2.0, breakdown=0.0,
)

def make_lp(p: float) -> Functional:
    """L^p mean: ( (1/s) sum |r|^p )^(1/p), Gaussian-normalized."""
    c = _lp_gauss_const(p)
    return Functional(
        f"lp{p:g}",
        lambda R, p=p, c=c: (np.mean(np.abs(R) ** p, axis=1)) ** (1.0 / p) / c,
        moment_requirement=p, breakdown=0.0,
    )

l1 = make_lp(1.0)

mad = Functional(
    "mad",
    lambda R: _MAD_CONST * np.median(
        np.abs(R - np.median(R, axis=1, keepdims=True)), axis=1),
    moment_requirement=0.0, breakdown=0.5,
)

def make_quantile(tau: float) -> Functional:
    """tau-quantile of |residuals|, Gaussian-normalized."""
    c = _norm_ppf(0.5 * (1.0 + tau))
    return Functional(
        f"q{tau:g}",
        lambda R, tau=tau, c=c: np.quantile(np.abs(R), tau, axis=1) / c,
        moment_requirement=0.0, breakdown=min(tau, 1.0 - tau) if tau != 0.5 else 0.5,
    )

iqr = Functional(
    "iqr",
    lambda R: (np.quantile(R, 0.75, axis=1) - np.quantile(R, 0.25, axis=1))
              / (2.0 * _norm_ppf(0.75)),
    moment_requirement=0.0, breakdown=0.25,
)

def make_huber(c: float = 1.345, n_iter: int = 30) -> Functional:
    """Huber Proposal-2 scale M-estimator, Gaussian-consistent."""
    # kappa = E[ min(Z^2, c^2) ] for Z ~ N(0,1)
    kappa = (math.erf(c / _SQRT2) - 2.0 * c * _phi_pdf(c)) \
            + c * c * (1.0 - math.erf(c / _SQRT2))

    def _huber(R: np.ndarray, c=c, kappa=kappa, n_iter=n_iter) -> np.ndarray:
        # initialize with MAD (robust start); fall back to tiny positive
        s = _MAD_CONST * np.median(
            np.abs(R - np.median(R, axis=1, keepdims=True)), axis=1)
        s = np.where(s <= 0, np.std(R, axis=1) + 1e-300, s)
        n = R.shape[1]
        for _ in range(n_iter):
            u = R / (s[:, None] + 1e-300)
            w = np.minimum(u ** 2, c ** 2)
            s = s * np.sqrt(np.sum(w, axis=1) / (n * kappa))
        return s

    return Functional(f"huber{c:g}", _huber,
                      moment_requirement=0.0, breakdown=0.5)

linf = Functional(
    "linf",
    lambda R: np.max(np.abs(R), axis=1),   # no finite Gaussian constant (grows ~ sqrt(2 log s))
    moment_requirement=0.0, breakdown=0.0,
)

def make_trimmed_std(alpha: float = 0.1) -> Functional:
    """Std of the central (1-2*alpha) fraction of |residuals| (symmetric trim).

    Approximate Gaussian consistency via truncated-normal second moment.
    """
    zc = _norm_ppf(1.0 - alpha)
    # E[Z^2 | |Z| <= zc] = 1 - 2 zc phi(zc) / (1 - 2 alpha)
    denom = 1.0 - 2.0 * alpha
    c2 = 1.0 - 2.0 * zc * _phi_pdf(zc) / denom
    c = math.sqrt(max(c2, 1e-12))

    def _tstd(R: np.ndarray, alpha=alpha, c=c) -> np.ndarray:
        A = np.sort(np.abs(R), axis=1)
        s = R.shape[1]
        k = int(math.floor((1.0 - alpha) * s))
        k = max(k, 2)
        return np.sqrt(np.mean(A[:, :k] ** 2, axis=1)) / c

    return Functional(f"tstd{alpha:g}", _tstd,
                      moment_requirement=0.0, breakdown=alpha)

entropy_power = Functional(
    # exp( differential entropy estimate ) is positively homogeneous of
    # degree 1  (h(aR) = h(R) + log a)  -> admissible after exponentiation.
    # Estimated via Vasicek spacings; Gaussian-normalized.
    "entpow",
    None,  # set below
    moment_requirement=0.0, breakdown=0.0,
)

def _entropy_power_fn(R: np.ndarray) -> np.ndarray:
    n = R.shape[1]
    m = max(1, int(round(math.sqrt(n))))
    S = np.sort(R, axis=1)
    upper = S[:, np.minimum(np.arange(n) + m, n - 1)]
    lower = S[:, np.maximum(np.arange(n) - m, 0)]
    spacings = np.maximum(upper - lower, 1e-300)
    h = np.mean(np.log(n / (2.0 * m) * spacings), axis=1)
    return np.exp(h) / _ENTROPY_GAUSS

object.__setattr__(entropy_power, "fn", _entropy_power_fn)


#: Default battery of functionals used by the validation script.
FUNCTIONALS: Dict[str, Functional] = {
    "rms":     rms,                    # classical DFA
    "l1":      l1,
    "lp0.5":   make_lp(0.5),
    "mad":     mad,
    "q0.75":   make_quantile(0.75),
    "q0.9":    make_quantile(0.9),
    "iqr":     iqr,
    "huber":   make_huber(1.345),
    "tstd10":  make_trimmed_std(0.10),
    "linf":    linf,
    "entpow":  entropy_power,
}


# ----------------------------------------------------------------------
# Core algorithm
# ----------------------------------------------------------------------

def default_scales(n: int, s_min: int = 16, s_max: int | None = None,
                   n_scales: int = 20) -> np.ndarray:
    """Logarithmically spaced integer scales."""
    if s_max is None:
        s_max = n // 4
    s = np.unique(np.round(np.exp(
        np.linspace(math.log(s_min), math.log(s_max), n_scales))).astype(int))
    return s[s >= s_min]


from functools import lru_cache


@lru_cache(maxsize=256)
def _hat_matrix(s: int, order: int) -> np.ndarray:
    """Projection matrix onto polynomials of degree <= order on {0..s-1}."""
    t = np.arange(s, dtype=float)
    t = (t - t.mean()) / max(t.std(), 1.0)   # conditioning
    X = np.vander(t, order + 1, increasing=True)
    Q, _ = np.linalg.qr(X)
    return Q @ Q.T


def _window_residuals(Y: np.ndarray, s: int, order: int) -> np.ndarray:
    """Detrending residuals for all windows of size s (forward + backward).

    Returns array of shape (2 * (n // s), s).
    """
    n = Y.size
    ns = n // s
    fwd = Y[: ns * s].reshape(ns, s)
    bwd = Y[n - ns * s:].reshape(ns, s)
    W = np.vstack([fwd, bwd])
    H = _hat_matrix(s, order)
    return W - W @ H  # H is symmetric: residual = (I - H) y per row


@dataclass
class FDFAResult:
    scales: np.ndarray
    F: np.ndarray                 # aggregated fluctuation per scale
    slope: float                  # scaling exponent (Hurst-type)
    intercept: float
    stderr: float
    functional: str
    order: int
    aggregate: str
    Fv: dict = field(default_factory=dict, repr=False)  # per-window values


def _aggregate(Fv: np.ndarray, how: str, q: float) -> float:
    Fv = Fv[np.isfinite(Fv) & (Fv > 0)]
    if Fv.size == 0:
        return np.nan
    if how == "qmean":
        if abs(q) < 1e-12:
            return float(np.exp(np.mean(np.log(Fv))))
        return float(np.mean(Fv ** q) ** (1.0 / q))
    if how == "median":
        return float(np.median(Fv))
    raise ValueError(f"unknown aggregate '{how}'")


def fdfa(x: Sequence[float], phi: Functional = rms, *,
         scales: np.ndarray | None = None, order: int = 1,
         aggregate: str = "qmean", q: float = 2.0,
         fit_range: tuple | None = None,
         keep_windows: bool = False) -> FDFAResult:
    """Functional DFA of the series x with local functional phi.

    Parameters
    ----------
    x : 1-D array-like, the (noise-like) input series.
    phi : Functional, local dispersion measure on window residuals.
    scales : integer window sizes (default: log-spaced 16 .. n/4).
    order : polynomial detrending order m (default 1, i.e. FDFA-1).
    aggregate : 'qmean' -> ( mean_v Fv^q )^(1/q)  (q=2 mimics classical DFA
                aggregation), or 'median' for a fully robust outer layer.
    q : outer aggregation exponent when aggregate='qmean'.
    fit_range : optional (s_lo, s_hi) restriction for the log-log fit.
    keep_windows : store per-window fluctuation values in result.Fv.

    Returns
    -------
    FDFAResult with fitted scaling exponent 'slope'.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    Y = np.cumsum(x - x.mean())
    n = Y.size
    if scales is None:
        scales = default_scales(n)
    scales = np.asarray(scales, dtype=int)
    scales = scales[(scales >= order + 2) & (scales <= n // 2)]

    F = np.empty(scales.size)
    Fv_store = {}
    for k, s in enumerate(scales):
        R = _window_residuals(Y, int(s), order)
        Fv = phi(R)
        F[k] = _aggregate(Fv, aggregate, q)
        if keep_windows:
            Fv_store[int(s)] = Fv

    good = np.isfinite(F) & (F > 0)
    ls, lF = np.log(scales[good]), np.log(F[good])
    if fit_range is not None:
        m = (scales[good] >= fit_range[0]) & (scales[good] <= fit_range[1])
        ls, lF = ls[m], lF[m]
    A = np.vstack([ls, np.ones_like(ls)]).T
    coef, res_, *_ = np.linalg.lstsq(A, lF, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    dof = max(ls.size - 2, 1)
    resid = lF - A @ coef
    stderr = float(np.sqrt(np.sum(resid ** 2) / dof /
                           np.sum((ls - ls.mean()) ** 2)))

    return FDFAResult(scales=scales, F=F, slope=slope, intercept=intercept,
                      stderr=stderr, functional=phi.name, order=order,
                      aggregate=aggregate, Fv=Fv_store)


def hurst(x, phi: Functional = rms, **kw) -> float:
    """Convenience: scaling exponent only."""
    return fdfa(x, phi, **kw).slope


def _fit_slope(scales: np.ndarray, F: np.ndarray,
               fit_range: tuple | None) -> tuple:
    good = np.isfinite(F) & (F > 0)
    ls, lF = np.log(scales[good]), np.log(F[good])
    if fit_range is not None:
        m = (scales[good] >= fit_range[0]) & (scales[good] <= fit_range[1])
        ls, lF = ls[m], lF[m]
    coef = np.polyfit(ls, lF, 1)
    return float(coef[0]), float(coef[1])


def fdfa_multi(x, specs, *, scales: np.ndarray | None = None,
               order: int = 1, fit_range: tuple | None = None
               ) -> Dict[str, FDFAResult]:
    """FDFA for many (functional, aggregator) pipelines sharing residuals.

    The detrending residuals are the expensive part and are identical for
    every pipeline; this computes them once per scale.

    Parameters
    ----------
    specs : iterable of (label, Functional, aggregate, q) tuples.

    Returns
    -------
    dict label -> FDFAResult.
    """
    specs = list(specs)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    Y = np.cumsum(x - x.mean())
    n = Y.size
    if scales is None:
        scales = default_scales(n)
    scales = np.asarray(scales, dtype=int)
    scales = scales[(scales >= order + 2) & (scales <= n // 2)]

    # unique functionals by name
    phis: Dict[str, Functional] = {}
    for _, phi, _, _ in specs:
        phis.setdefault(phi.name, phi)

    F = {lab: np.empty(scales.size) for lab, *_ in specs}
    for k, s in enumerate(scales):
        R = _window_residuals(Y, int(s), order)
        Fv = {name: phi(R) for name, phi in phis.items()}
        for lab, phi, agg, q in specs:
            F[lab][k] = _aggregate(Fv[phi.name], agg, q)

    out = {}
    for lab, phi, agg, q in specs:
        slope, intercept = _fit_slope(scales, F[lab], fit_range)
        out[lab] = FDFAResult(scales=scales, F=F[lab], slope=slope,
                              intercept=intercept, stderr=float("nan"),
                              functional=phi.name, order=order,
                              aggregate=agg)
    return out


def mfdfa_hq_multi(x, qs: Sequence[float], phis, *,
                   scales: np.ndarray | None = None, order: int = 1,
                   fit_range: tuple | None = None) -> Dict[str, HqResult]:
    """Generalized MFDFA h(q) for several inner functionals, sharing
    the detrending residuals.

    phis : dict {label: Functional} or sequence of Functionals
           (then keyed by phi.name). Returns dict label -> HqResult.
    """
    if not isinstance(phis, dict):
        phis = {phi.name: phi for phi in phis}
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    Y = np.cumsum(x - x.mean())
    n = Y.size
    if scales is None:
        scales = default_scales(n)
    scales = np.asarray(scales, dtype=int)
    scales = scales[(scales >= order + 2) & (scales <= n // 2)]
    qs = np.asarray(qs, dtype=float)

    Fq = {lab: np.empty((qs.size, scales.size)) for lab in phis}
    for k, s in enumerate(scales):
        R = _window_residuals(Y, int(s), order)
        for lab, phi in phis.items():
            Fv = phi(R)
            Fv = Fv[np.isfinite(Fv) & (Fv > 0)]
            for j, q in enumerate(qs):
                Fq[lab][j, k] = _aggregate(Fv, "qmean", q)

    out = {}
    for lab, phi in phis.items():
        hq = np.empty(qs.size)
        for j in range(qs.size):
            hq[j] = _fit_slope(scales, Fq[lab][j], fit_range)[0]
        out[lab] = HqResult(qs=qs, hq=hq, tq=qs * hq - 1.0,
                            scales=scales, Fq=Fq[lab],
                            functional=phi.name)
    return out


# ----------------------------------------------------------------------
# Multifractal layer: generalized h(q)
# ----------------------------------------------------------------------

@dataclass
class HqResult:
    qs: np.ndarray
    hq: np.ndarray
    tq: np.ndarray               # tau(q) = q h(q) - 1
    scales: np.ndarray
    Fq: np.ndarray               # shape (len(qs), len(scales))
    functional: str


def mfdfa_hq(x, qs: Sequence[float], phi: Functional = rms, *,
             scales: np.ndarray | None = None, order: int = 1,
             fit_range: tuple | None = None) -> HqResult:
    """Generalized MFDFA: local functional phi, outer q-moments.

    Classical MFDFA is phi=rms. Note the two-layer structure: phi decides
    *what* is measured inside a window; q decides *which windows dominate*.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    Y = np.cumsum(x - x.mean())
    n = Y.size
    if scales is None:
        scales = default_scales(n)
    scales = np.asarray(scales, dtype=int)
    scales = scales[(scales >= order + 2) & (scales <= n // 2)]
    qs = np.asarray(qs, dtype=float)

    Fq = np.empty((qs.size, scales.size))
    for k, s in enumerate(scales):
        R = _window_residuals(Y, int(s), order)
        Fv = phi(R)
        Fv = Fv[np.isfinite(Fv) & (Fv > 0)]
        for j, q in enumerate(qs):
            Fq[j, k] = _aggregate(Fv, "qmean", q)

    hq = np.empty(qs.size)
    for j in range(qs.size):
        good = np.isfinite(Fq[j]) & (Fq[j] > 0)
        ls, lF = np.log(scales[good]), np.log(Fq[j, good])
        if fit_range is not None:
            m = (scales[good] >= fit_range[0]) & (scales[good] <= fit_range[1])
            ls, lF = ls[m], lF[m]
        hq[j] = np.polyfit(ls, lF, 1)[0]

    tq = qs * hq - 1.0
    return HqResult(qs=qs, hq=hq, tq=tq, scales=scales, Fq=Fq,
                    functional=phi.name)
