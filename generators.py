"""
generators.py — Synthetic benchmark processes for FDFA validation
==================================================================

Ground-truth processes with known scaling exponents:

  * fgn(n, H)              fractional Gaussian noise  -> DFA exponent H
  * alpha_stable(n, alpha) iid symmetric alpha-stable -> DFA exponent 1/alpha
  * binomial_cascade(J,m0) multiplicative cascade     -> known h(q):
        h(q) = 1/q - log2(m0^q + (1-m0)^q)/q     (q != 0)

All generators take an optional numpy Generator (rng) for reproducibility.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["fgn", "alpha_stable", "binomial_cascade", "cascade_hq_theory",
           "mrw", "mrw_hq_theory", "arfima", "tempered_stable"]


def fgn(n: int, H: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Exact fractional Gaussian noise via Davies-Harte circulant embedding.

    Returns a length-n sample of fGn with Hurst exponent H, unit variance.
    DFA applied to this series should yield scaling exponent ~ H.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0, 1)")
    if abs(H - 0.5) < 1e-12:
        return rng.standard_normal(n)

    def gamma(k: np.ndarray) -> np.ndarray:
        k = np.abs(k).astype(float)
        return 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H)
                      + np.abs(k - 1) ** (2 * H))

    m = n
    g = gamma(np.arange(m + 1))
    # circulant first row: [g0 .. gm, g_{m-1} .. g_1], length 2m
    row = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(row).real
    lam = np.maximum(lam, 0.0)           # clip tiny negative eigenvalues

    n2 = row.size                        # = 2m
    Z = rng.standard_normal(n2) + 1j * rng.standard_normal(n2)
    X = np.fft.fft(np.sqrt(lam / n2) * Z)
    return X[:n].real


def alpha_stable(n: int, alpha: float, beta: float = 0.0, scale: float = 1.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """iid alpha-stable noise via Chambers-Mallows-Stuck.

    For symmetric (beta=0) stable noise the integrated walk is 1/alpha
    self-similar, so the DFA exponent is 1/alpha. For alpha < 2 the
    variance is infinite: variance-based DFA estimators become erratic
    while quantile/median-based functionals remain stable.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha must be in (0, 2]")
    if abs(alpha - 2.0) < 1e-12:
        return math.sqrt(2.0) * scale * rng.standard_normal(n)

    V = rng.uniform(-math.pi / 2.0, math.pi / 2.0, size=n)
    W = rng.exponential(1.0, size=n)

    if abs(alpha - 1.0) < 1e-8:
        alpha = 1.0 + 1e-8  # avoid the alpha == 1 branch singularity

    zeta = -beta * math.tan(math.pi * alpha / 2.0)
    xi = math.atan(-zeta) / alpha
    num = np.sin(alpha * (V + xi))
    den = (np.cos(V)) ** (1.0 / alpha)
    tail = (np.cos(V - alpha * (V + xi)) / W) ** ((1.0 - alpha) / alpha)
    X = (1.0 + zeta ** 2) ** (1.0 / (2.0 * alpha)) * (num / den) * tail
    return scale * X


def binomial_cascade(J: int, m0: float = 0.6,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Random binomial multiplicative cascade on 2**J points.

    At each dyadic split, mass fractions (m0, 1-m0) are assigned to the
    two halves in random order. Returns the resulting measure density
    (length 2**J), a canonical multifractal benchmark.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0.0 < m0 < 1.0):
        raise ValueError("m0 must be in (0, 1)")
    mu = np.ones(1)
    for _ in range(J):
        left = np.where(rng.random(mu.size) < 0.5, m0, 1.0 - m0)
        right = 1.0 - left
        mu = np.column_stack([mu * left, mu * right]).ravel()
    # normalize to unit mean so the series is a density-like signal
    return mu * mu.size


def _gaussian_circulant(cov_row: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Sample a stationary Gaussian vector with autocovariance cov_row
    (cov_row[k] = Cov(X_0, X_k), k = 0..n) via circulant embedding."""
    n = cov_row.size - 1
    row = np.concatenate([cov_row, cov_row[-2:0:-1]])
    lam = np.maximum(np.fft.fft(row).real, 0.0)
    n2 = row.size
    Z = rng.standard_normal(n2) + 1j * rng.standard_normal(n2)
    X = np.fft.fft(np.sqrt(lam / n2) * Z)
    return X[:n].real


def mrw(n: int, lam2: float = 0.05, L: int | None = None,
        rng: np.random.Generator | None = None) -> np.ndarray:
    """Multifractal Random Walk increments (Bacry-Delour-Muzy, H=1/2).

    x_k = eps_k * exp(omega_k), with eps iid N(0,1) and omega Gaussian,
    Cov(omega_i, omega_j) = lam2 * ln( L / (1+|i-j|) )  for |i-j| < L.

    Genuine (log-normal) multifractality with *Gaussian* tails:
        zeta(q) = (1/2 + lam2) q - (lam2/2) q^2,
        h(q)    = 1/2 + lam2 - (lam2/2) q          (so h(2) = 1/2).
    """
    if rng is None:
        rng = np.random.default_rng()
    if L is None:
        L = n
    k = np.arange(n + 1, dtype=float)
    cov = lam2 * np.log(np.maximum(L / (1.0 + k), 1.0))
    omega = _gaussian_circulant(cov, rng)
    omega -= cov[0]          # normalization; irrelevant for exponents
    eps = rng.standard_normal(n)
    return eps * np.exp(omega)


def mrw_hq_theory(qs: np.ndarray, lam2: float = 0.05) -> np.ndarray:
    """h(q) = 1/2 + lam2 - (lam2/2) q for the MRW above."""
    qs = np.asarray(qs, dtype=float)
    return 0.5 + lam2 - 0.5 * lam2 * qs


def arfima(n: int, d: float, rng: np.random.Generator | None = None,
           burn: int | None = None) -> np.ndarray:
    """ARFIMA(0, d, 0): fractionally integrated Gaussian white noise.

    MA(infinity) representation truncated at n + burn coefficients,
    convolved by FFT. Long-memory parameter d in (-0.5, 0.5);
    DFA scaling exponent = d + 1/2.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (-0.5 < d < 0.5):
        raise ValueError("d must be in (-0.5, 0.5)")
    if burn is None:
        burn = n
    m = n + burn
    psi = np.empty(m)
    psi[0] = 1.0
    for k in range(1, m):
        psi[k] = psi[k - 1] * (k - 1.0 + d) / k
    eps = rng.standard_normal(m)
    nfft = 1 << int(math.ceil(math.log2(2 * m - 1)))
    x = np.fft.irfft(np.fft.rfft(psi, nfft) * np.fft.rfft(eps, nfft),
                     nfft)[:m]
    return x[burn:]


def tempered_stable(n: int, alpha: float, ell: float = 10.0,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Truncated (exponentially tempered) Levy noise.

    Symmetric alpha-stable increments accepted with probability
    exp(-|x|/ell): heavy-tailed at small |x|, finite variance overall.
    Exhibits the classic truncated-Levy crossover (stable-like at small
    scales, Gaussian at large scales).
    """
    if rng is None:
        rng = np.random.default_rng()
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = alpha_stable(2 * (n - filled), alpha, rng=rng)
        acc = cand[rng.random(cand.size) < np.exp(-np.abs(cand) / ell)]
        take = min(acc.size, n - filled)
        out[filled:filled + take] = acc[:take]
        filled += take
    return out


def cascade_hq_theory(qs: np.ndarray, m0: float = 0.6) -> np.ndarray:
    """Theoretical generalized Hurst exponent of the binomial cascade.

    tau(q) = -log2(m0^q + (1-m0)^q);  h(q) = (tau(q) + 1)/q  for q != 0.
    (Convention: MFDFA applied to the cascade *series* itself.)
    """
    qs = np.asarray(qs, dtype=float)
    m1 = 1.0 - m0
    hq = np.empty(qs.size)
    for i, q in enumerate(qs):
        if abs(q) < 1e-9:
            # limit q -> 0
            hq[i] = -0.5 * (math.log2(m0) + math.log2(m1))
        else:
            tau = -math.log2(m0 ** q + m1 ** q)
            hq[i] = (tau + 1.0) / q
    return hq
