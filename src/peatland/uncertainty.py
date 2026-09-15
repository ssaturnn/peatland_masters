"""Stratified bootstrap intervals for complete independently reviewed samples."""

from collections import defaultdict
import numpy as np
from .evaluation import METHODS, _prediction, score_points

VERSION = '2026-09-15-stratified-bootstrap-v1'


def rescaled_stratum(probabilities, sample_size, population, replicates, rng):
    """Resample n-1 observations and apply the finite-population correction."""
    p = np.asarray(probabilities, float)
    if (sample_size < 1 or population < sample_size or replicates < 1
            or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1)):
        raise ValueError('Invalid stratum probabilities or sizes')
    if sample_size == population:
        return np.tile(p, (replicates, 1))
    if sample_size == 1:
        raise ValueError('A non-census singleton stratum cannot estimate uncertainty')
    draws = rng.multinomial(sample_size - 1, p, size=replicates) / (sample_size - 1)
    return p + np.sqrt(1 - sample_size / population) * (draws - p)


def bootstrap_intervals(rows, method_names=METHODS, replicates=2000, confidence=.95, seed=42):
    """Preserve site-year strata and N/n weights in every bootstrap replicate."""
    if not isinstance(replicates, int) or replicates < 100 or not 0 < confidence < 1:
        raise ValueError('Use at least 100 replicates and a confidence level in (0, 1)')
    report = score_points(rows, method_names=method_names)
    result = dict(version=VERSION, confidence=confidence, replicates=replicates, seed=seed,
                  resampling='n-1 within each site/year/stratum; finite-population rescaling',
                  status='unavailable', methods={})
    if not report['weighting_available']:
        result['reason'] = 'Complete labels and valid stratum population/sample sizes are required'
        return result
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get('site'), row.get('year'), row.get('stratum'))].append(row)
    if any(len(g) == 1 and float(g[0]['stratum_population']) > 1 for g in groups.values()):
        result['reason'] = 'A non-census singleton stratum has unknown sampling variance'
        return result
    tails = [(1-confidence)/2, 1-(1-confidence)/2]
    for method in method_names:
        if method not in report['methods'] or report['methods'][method]['weighted'] is None:
            continue
        rng = np.random.default_rng(seed)
        confusion = np.zeros((replicates, 4), float)
        for group in groups.values():
            cells = []
            for row in group:
                actual = row['truth'].strip().lower() == 'bare_peat'
                predicted = bool(_prediction(row, method))
                cells.append((0 if actual else 1) if predicted else (2 if actual else 3))
            p = np.bincount(cells, minlength=4) / len(group)
            population = int(float(group[0]['stratum_population']))
            confusion += population * rescaled_stratum(p, len(group), population, replicates, rng)
        tp, fp, fn, tn = confusion.T
        def ratio(a, b):
            return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
        values = dict(precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                      f1=ratio(2*tp, 2*tp+fp+fn), accuracy=ratio(tp+tn, tp+fp+fn+tn),
                      reference_bare_fraction=ratio(tp+fn, tp+fp+fn+tn),
                      reference_bare_pixels=tp+fn)
        intervals = {}
        for metric, samples in values.items():
            finite = samples[np.isfinite(samples)]
            bounds = np.quantile(finite, tails).tolist() if len(finite) else [None, None]
            intervals[metric] = dict(lower=bounds[0], upper=bounds[1],
                                     defined_replicates=len(finite), undefined_replicates=replicates-len(finite))
        result['methods'][method] = intervals
    result['status'] = 'complete' if result['methods'] else 'unavailable'
    result['limitations'] = [
        'Conditional on sampled site-years; does not include between-site or label uncertainty.',
        'Requires simple random pixel sampling without replacement within recorded strata.',
        'Spatial dependence and unsampled strata are not estimated.',
        'Percentiles for ratios use defined replicates only; inspect undefined counts.',
        'Homogeneous observed strata give zero bootstrap variance; rare unseen errors remain possible.']
    return result
