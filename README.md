# FDFA — Functional Detrended Fluctuation Analysis

A two-layer generalization of DFA/MFDFA in which the local fluctuation
measure (classically, the variance of the detrending residuals) is a
pluggable **dispersion functional** Φ, and the aggregation across
windows is a pluggable **outer layer** (power means or the median).
Classical DFA is the member FDFA(RMS, A₂); classical MFDFA is the
slice {FDFA(RMS, A_q)}.

Why bother:

- **Exponent universality.** Every admissible Φ (homogeneous,
  symmetric, sampling-consistent) yields the *same* scaling exponent
  on self-similar inputs — so swapping the variance for a robust
  functional never changes the target, only the estimator statistics.
- **Heavy tails.** Classical DFA is asymptotically blind to the
  self-similarity index of stable noise (it returns h(2)=1/2 for
  *every* α<2). A median outer layer recovers 1/α with any inner Φ.
- **Trend absorption.** Robustness *inside* the window cannot reject
  integrated extremes (they become window-wide trends, not outliers);
  robustness only works at the cross-window layer. This came out of a
  falsified prediction — see the ledger in `docs/`.
- **A surrogate-free discriminator.** The quantile-exponent profile
  θ(τ) — slopes of cross-window quantiles of F(v,s) — is flat when a
  wide h(q) spectrum is of purely distributional (heavy-tail) origin
  and τ-dependent when the multifractality is genuine. Its verdict
  matches the standard shuffle-surrogate test on five classes of
  synthetic benchmarks and on real heliophysical data, at a fraction
  of the cost.

The reasoning and the full prediction–falsification record are in
`docs/fdfa_theory_note.pdf`.

## Installation

```bash
pip install numpy matplotlib
pip install scienceplots   # optional, for publication-style figures
```

Python ≥ 3.10. No other dependencies.

## Quick start

```python
import numpy as np
from fdfa import fdfa, mfdfa_hq, FUNCTIONALS
from generators import fgn, alpha_stable

# a robust Hurst estimate (Huber inner functional, median outer layer)
x = fgn(2**14, H=0.7, rng=np.random.default_rng(0))
res = fdfa(x, FUNCTIONALS["huber"], order=2, aggregate="median")
print(res.slope)          # ~0.70

# classical DFA is one member of the family
res = fdfa(x, FUNCTIONALS["rms"], order=2, aggregate="qmean", q=2.0)

# generalized h(q) with any inner functional
hq = mfdfa_hq(x, qs=np.linspace(-4, 4, 17), phi=FUNCTIONALS["rms"])

# where classical DFA fails: iid stable noise, true exponent 1/1.5
y = alpha_stable(2**14, alpha=1.5, rng=np.random.default_rng(0))
print(fdfa(y, FUNCTIONALS["rms"], aggregate="qmean", q=2.0).slope)  # ~0.5 (blind)
print(fdfa(y, FUNCTIONALS["rms"], aggregate="median").slope)        # ~0.667
```

Available inner functionals (`FUNCTIONALS` dict): `rms`, `l1`,
`lp0.5`, `mad`, `q0.75`, `q0.9`, `iqr`, `huber`, `tstd10`, `linf`,
`entpow` — all normalized to Gaussian σ-consistency. Factories
`make_lp(p)`, `make_quantile(tau)`, `make_huber(c)`,
`make_trimmed_std(alpha)` build others.

## Reproducing the results

```bash
python run_validation.py            # 9 synthetic experiments, ~1 min
python run_validation.py --quick    # fast sanity check
python run_validation.py --only 5   # a single experiment (1-9)
python run_application.py           # OMNI2 heliophysical analysis
```

Figures, CSV tables and text summaries are written to `results/`
(the committed contents of that folder were produced by exactly these
commands, seed 42).

For `run_application.py`, download the OMNI2 hourly subset (scalar B,
sunspot number R, Dst, F10.7; 1976–present) from
[NASA/GSFC OMNIWeb](https://omniweb.gsfc.nasa.gov/form/dx1.html) and
place it at `real_data/omni2_<id>.lst.txt`; the expected column
layout is documented in `data/omni2_a7kZN7QddQ.fmt.txt`.

## Repository layout

```
fdfa.py               core estimator: functionals + two-layer FDFA + h(q)
generators.py         benchmarks with known ground truth
                      (fGn, ARFIMA, stable, tempered stable, cascade, MRW)
run_validation.py     the nine-experiment prediction/falsification suite
run_application.py    OMNI2 application (sunspot, Dst, IMF B, δB)
docs/                 working notes (PDF + LaTeX source)
results/              committed outputs of the two scripts
data/                 OMNI2 format file + download instructions
```

## Citation

A manuscript describing the framework is in preparation. Until it is
available, please cite this repository.

## License

MIT — see `LICENSE`.
