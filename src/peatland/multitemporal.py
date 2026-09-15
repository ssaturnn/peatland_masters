"""Versioned seasonal-trajectory candidates and tidal-buffer diagnostics."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from .gng import GrowingNeuralGas

VERSION = "2026-09-15-multitemporal-gng-v2"
TIDAL_VERSION = "2026-09-15-tidal-buffer-v1"
FEATURES = ('early_ndvi', 'early_nbr', 'early_swir1', 'early_mndwi',
            'late_ndvi', 'late_nbr', 'late_swir1', 'late_mndwi',
            'delta_ndvi', 'delta_nbr', 'delta_swir1', 'delta_mndwi')


@dataclass(frozen=True)
class Parameters:
    ndvi_cap: float = .25
    ndvi_floor: float = .10
    contrast_drop: float = .18
    max_greenup: float = .10
    swir_floor: float = .07
    nbr_floor: float = 0.
    brightness_cap: float = .30
    max_nodes: int = 80
    training_pixels: int = 8000
    training_steps: int = 15000
    seed: int = 42

    def as_dict(self):
        return asdict(self)


def _indices(stack):
    """NDVI, NBR, SWIR1 and MNDWI of an H x W x 6 reflectance*10000 stack."""
    blue, green, red, nir, swir1, swir2 = np.moveaxis(np.asarray(stack, float) / 10000, -1, 0)
    ratio = lambda a, b: (a - b) / (a + b + 1e-9)
    with np.errstate(invalid='ignore'):
        return np.stack((ratio(nir, red), ratio(nir, swir2), swir1,
                         ratio(green, swir1)), axis=-1)


def _clear(stack, valid, inside):
    return (np.asarray(valid, bool) & np.asarray(inside, bool)
            & np.isfinite(np.asarray(stack, float)).all(axis=-1))


def trajectory_features(early, late):
    """NDVI, NBR, SWIR1 and MNDWI at two dates, followed by late minus early."""
    early, late = np.asarray(early, float), np.asarray(late, float)
    if early.shape != late.shape or early.ndim != 3 or early.shape[-1] != 6:
        raise ValueError('Expected matching H x W x 6 image stacks')
    a, b = _indices(early), _indices(late)
    return np.concatenate((a, b, b - a), axis=-1)


def adaptive_threshold(ndvi, params):
    """Bound a site/date threshold using only finite comparison-support pixels."""
    values = np.asarray(ndvi)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        return params.ndvi_cap
    return min(params.ndvi_cap, max(params.ndvi_floor,
               float(np.quantile(values, .75)) - params.contrast_drop))


def early_candidates(early, valid_early, inside, params=None):
    """Single-date candidates on the early scene's own clear bog pixels.

    The same physical guards as the persistence gate, on one date: these are
    the pixels a late scene must see for the persistence test to say anything.
    """
    params = params or Parameters()
    early = np.asarray(early, float)
    clear = _clear(early, valid_early, inside)
    x = _indices(early)
    threshold = adaptive_threshold(x[..., 0][clear], params)
    with np.errstate(invalid='ignore'):
        mask = (clear & (x[..., 0] < threshold) & (x[..., 1] > params.nbr_floor)
                & (x[..., 2] >= params.swir_floor) & (x[..., 3] <= 0)
                & (early[..., :3].mean(axis=-1) / 10000 < params.brightness_cap))
    return mask, threshold


def candidate_cover(candidates, valid_late):
    """Share of the early candidates that the late scene sees clearly.

    A diagnostic, not an acceptance rule: late scenes pass the v3 whole-bog
    gates, and unseen candidates stay outside the comparison support.
    """
    cand = np.asarray(candidates, bool)
    n = int(cand.sum())
    m = int((cand & np.asarray(valid_late, bool)).sum())
    return {'candidate_pixels': n, 'seen_pixels': m, 'cover': m / n if n else None}


def physical_gate(features, early_threshold, late_threshold, params):
    """Require persistent low vegetation, nonwater and positive NBR at both dates."""
    x = np.asarray(features)
    return ((x[..., 0] < early_threshold) & (x[..., 4] < late_threshold)
            & (x[..., 8] <= params.max_greenup)
            & (x[..., 1] > params.nbr_floor) & (x[..., 5] > params.nbr_floor)
            & (x[..., 2] >= params.swir_floor) & (x[..., 6] >= params.swir_floor)
            & (x[..., 3] <= 0) & (x[..., 7] <= 0))


def detect_trajectory(early, late, valid_early, valid_late, inside, params=None):
    """Return rules and GNG masks on common clear support; unknown stays explicit.

    Each date's adaptive NDVI threshold comes from that date's own clear bog
    pixels, so a partly clouded late scene cannot shift the early threshold.
    """
    params = params or Parameters()
    if params.max_nodes < 2 or params.training_steps < 1 or params.training_pixels < 2:
        raise ValueError('Invalid GNG training parameters')
    features = trajectory_features(early, late)
    shape = features.shape[:2]
    if any(np.asarray(m).shape != shape for m in (valid_early, valid_late, inside)):
        raise ValueError('Masks must match the image grid')
    early_rule, early_t = early_candidates(early, valid_early, inside, params)
    late_clear = _clear(late, valid_late, inside)
    late_t = adaptive_threshold(_indices(late)[..., 0][late_clear], params)
    support = (_clear(early, valid_early, inside) & late_clear
               & np.isfinite(features).all(axis=-1))
    rules, gng = np.zeros(shape, bool), np.zeros(shape, bool)
    prototype_labels = np.full(shape, -1, np.int16)
    diagnostics = {'version': VERSION, 'support_pixels': int(support.sum()),
                   'early_candidate_pixels': int(early_rule.sum()),
                   'early_candidates_seen_late': int((early_rule & support).sum()),
                   'nodes': 0, 'bare_nodes': 0,
                   'early_threshold': early_t, 'late_threshold': late_t}
    if not support.any():
        return dict(rules=rules, gng=gng, early_rule=early_rule, support=support,
                    labels=prototype_labels, diagnostics=diagnostics)
    raw = features[support]
    bright = ((np.asarray(early)[support, :3].mean(axis=-1) / 10000 < params.brightness_cap)
              & (np.asarray(late)[support, :3].mean(axis=-1) / 10000 < params.brightness_cap))
    candidates = physical_gate(raw, early_t, late_t, params) & bright
    rules[support] = candidates
    mu, sd = raw.mean(axis=0), raw.std(axis=0) + 1e-9
    scaled = (raw - mu) / sd
    if len(raw) >= 2:
        rng = np.random.default_rng(params.seed)
        idx = rng.choice(len(raw), min(params.training_pixels, len(raw)), replace=False)
        net = GrowingNeuralGas(max_nodes=params.max_nodes, rng=rng)
        net.fit(scaled[idx], n_steps=params.training_steps)
        labels = net.predict(scaled)
        prototypes = net.weights * sd + mu
        bare_nodes = physical_gate(prototypes, early_t, late_t, params)
        gng[support] = candidates & bare_nodes[labels]
        prototype_labels[support] = labels
        diagnostics.update(nodes=len(prototypes), bare_nodes=int(bare_nodes.sum()),
                           prototypes=prototypes.tolist(), feature_mean=mu.tolist(),
                           feature_sd=sd.tolist())
    else:
        gng[support] = candidates
    return dict(rules=rules, gng=gng, early_rule=early_rule, support=support,
                labels=prototype_labels, diagnostics=diagnostics)


def tidal_buffer(water, pixels):
    """Square 10 m grid buffer including diagonal neighbours; zero is unchanged."""
    if not isinstance(pixels, int) or pixels < 0:
        raise ValueError('Buffer pixels must be a nonnegative integer')
    mask = np.asarray(water, bool)
    if mask.ndim != 2:
        raise ValueError('Expected a two-dimensional water mask')
    if pixels == 0:
        return mask.copy()
    return ndimage.binary_dilation(mask, structure=np.ones((2 * pixels + 1,) * 2, bool))


def buffered_inside(inside, water, pixels):
    """Detection area after excluding the ever-water mask grown by `pixels`."""
    inside = np.asarray(inside, bool)
    if inside.shape != np.asarray(water).shape:
        raise ValueError('Site and water masks must share a grid')
    return inside & ~tidal_buffer(water, pixels)


def mask_comparison(reference, candidate, support, pixel_ha=.01):
    """Areas and agreement on common support, without reference-label scoring."""
    reference, candidate, support = (np.asarray(m, bool) for m in (reference, candidate, support))
    if reference.shape != candidate.shape or reference.shape != support.shape:
        raise ValueError('Masks must share a grid')
    a, b = reference & support, candidate & support
    union = int((a | b).sum())
    n = int(support.sum())
    return dict(support_ha=n * pixel_ha, reference_ha=int(a.sum()) * pixel_ha,
                candidate_ha=int(b.sum()) * pixel_ha,
                removed_ha=int((a & ~b).sum()) * pixel_ha,
                added_ha=int((b & ~a).sum()) * pixel_ha,
                intersection_ha=int((a & b).sum()) * pixel_ha,
                iou=float((a & b).sum() / union) if union else None,
                agreement=float(((a == b) & support).sum() / n) if n else None)
