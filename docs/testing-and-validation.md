# Testing and validation

How correctness is checked at each level — from pure functions up to the
scientific claim that a detected patch is really cut peat.

## 1. Unit tests (deterministic, no network)

`tests/` — run with `python3 -m pytest tests/ -q`. These pin the maths
and logic that everything else depends on:

| File | Checks |
|---|---|
| `test_geo.py` | pixel area (10 m → 0.01 ha), mask-area summation, empty mask |
| `test_detect.py` | NDVI formula + divide-by-zero guard, range in [-1,1], threshold mask |
| `test_change_preprocess.py` | change classes (newly-bare / re-vegetated / stable), both-valid gating, SCL cloud mask, clear-fraction, designation labels, cutting-rate slope |
| `test_gng.py` | GNG grows > 2 nodes, is deterministic for a fixed seed, edge-pruning splits two separated blobs, `predict` labels every row |

They use tiny synthetic arrays, so they are fast and independent of
imagery or the network.

## 2. Data-integrity validation (generated dataset)

`scripts/validate_dataset.py` reads the produced `web/data/sites.geojson`
and asserts invariants that must hold for the map to be trustworthy:

- every feature carries the required properties;
- `ndvi_series` / `gng_series` lengths match `years`;
- bare area never exceeds the protected site area;
- coverage is a valid percentage; change areas are non-negative;
- **area cross-check:** the imagery-derived site area matches the NPWS
  register within 8% (a georeferencing / reprojection sanity check).

It exits non-zero on any failure, so it can gate a deploy.

## 3. Method validation (the scientific claim)

Unit tests prove the code is correct; they do **not** prove a detected
patch is genuinely cut peat. That is an accuracy question, addressed by:

- **Area cross-check** (already automated): imagery site area vs the NPWS
  register — confirms the geometry/area chain is right.
- **Two-detector agreement:** GNG vs the NDVI baseline are compared per
  site; large disagreement flags a site to inspect.
- **Cloud-aware change:** only pixels clear in both epochs are compared,
  so cloud cannot masquerade as change.
- **Point-based accuracy assessment (planned):** sample a few hundred
  random points across sites, label each by eye against high-resolution
  imagery, and report precision / recall / a confusion matrix. The method
  is unsupervised, so no training labels are needed — labelling is for
  evaluation only.

## 4. Web app checks

- `node --check web/app.js` — JavaScript syntax gate.
- Headless-Chrome screenshot of the running site — confirms the sidebar,
  ranking, sparkline and map render (the sidebar renders independently of
  the map's WebGL init, so ranking/detail work even if the map is slow).

## Quick command

```bash
python3 -m pytest tests/ -q            # unit tests
python3 scripts/validate_dataset.py    # dataset invariants + area cross-check
node --check web/app.js                # web syntax
```
