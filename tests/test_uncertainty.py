"""Analytical and synthetic bootstrap checks."""

import numpy as np
import pytest
from peatland.uncertainty import rescaled_stratum, bootstrap_intervals


def rows():
    result = []
    for s, population in [('a', 10), ('b', 90)]:
        for i in range(2):
            result.append(dict(point_id=f'{s}{i}', site='Synthetic', year='2022', stratum=s,
                               truth='bare_peat' if i == 0 else 'vegetated',
                               prediction_gng='1' if s == 'a' else '0',
                               stratum_population=str(population), stratum_sample_size='2'))
    return result


def test_rescaled_variance_matches_stratified_finite_population_formula():
    p = np.array([.3, .7])
    r = rescaled_stratum(p, 10, 100, 100000, np.random.default_rng(42))
    expected = (1-10/100) * (.3*.7*10/9) / 10
    assert r[:, 0].var() == pytest.approx(expected, rel=.02)
    assert np.all(r >= 0)
    assert np.allclose(r.sum(axis=1), 1)


def test_census_has_no_sampling_variance():
    r = rescaled_stratum([.5, .5], 2, 2, 100, np.random.default_rng(42))
    assert np.all(r == .5)


def test_weighted_intervals_are_reproducible_and_not_unweighted():
    a = bootstrap_intervals(rows(), replicates=500)
    assert a == bootstrap_intervals(rows(), replicates=500)
    r = a['methods']['gng']['recall']
    assert 0 <= r['lower'] <= .1 <= r['upper'] <= 1
    assert a['status'] == 'complete'


def test_missing_labels_and_non_census_singleton_block_intervals():
    sample = rows(); sample[0]['truth'] = ''
    assert bootstrap_intervals(sample, replicates=100)['status'] == 'unavailable'
    sample = rows()[:1]; sample[0]['stratum_sample_size'] = '1'
    assert 'singleton' in bootstrap_intervals(sample, replicates=100)['reason']


def test_undefined_precision_stays_null():
    sample = rows()
    for r in sample: r['prediction_gng'] = '0'
    result = bootstrap_intervals(sample, replicates=100)['methods']['gng']['precision']
    assert result['lower'] is None
    assert result['undefined_replicates'] == 100
