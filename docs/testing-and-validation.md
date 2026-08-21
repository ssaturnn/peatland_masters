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

## 3a. External validation against documented cutting (result)

The detector was cross-checked against the only public "ground truth" that
exists: the Department of Housing / NPWS records of turf plots cut without
consent in raised-bog SACs, reported by *thejournal.ie* and the *Irish
Times*. The three SACs with published 2022 plot counts were run through the
unsupervised pipeline **with no labels**:

| SAC (2022 record)      | Documented plots | Detected new bare peat 2018→2024 |
|------------------------|------------------|----------------------------------|
| Monivea (most-cut SAC) | 49               | 19.2 ha                          |
| Barroughter            | 42               | 2.9 ha                           |
| Callow                 | 31               | 8.6 ha                           |

All three documented hotspots are flagged as actively cutting, and
**Monivea — the most-cut SAC on record — is by a wide margin the largest
detection**, so the method clearly tracks real activity without any manual
annotation. The exact ordering of the two smaller sites does not match the
plot counts (Barroughter has more *plots* but less detected *area* than
Callow): plot count and hectares are different measures, the records are for
2022 while the detection is a 2018→2024 change, and n = 3 is small. So this
is corroboration of the signal, not a precision figure; a proper accuracy
assessment (§3) still needs plot-level or point-level reference data.

**Where GNG beats the NDVI baseline — a worked example.** At Rosroe Bog SAC
(2018) the NDVI baseline flags 1.2 ha of "bare peat" that is in fact the
site's open lough (SWIR ≈ 0.02, NIR ≈ 0.00 — water, not peat). GNG rejects
it via the SWIR signature and reports 0 ha, while still matching NDVI on the
real bare peat in later years. This is the detector's designed contribution:
NDVI-level sensitivity, with the full spectrum removing the water and shadow
a vegetation index cannot tell apart from bare peat.

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
