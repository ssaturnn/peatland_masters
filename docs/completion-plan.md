# Final-month completion plan

Updated 9 September 2026. The approved topic is satellite detection of candidate
turf-cutting / bare-peat surfaces in protected Irish bogs. The remaining time is
approximately one month, as confirmed by the author. The topic is fixed.

## Scope for submission

Deliver a reproducible Sentinel-2 pipeline, NDVI / spectral-rules / K-means / GNG
comparison, an independent reference assessment and a web map whose images match
its numbers. Describe detections as candidate bare surfaces: neither a bare pixel
nor protected-area membership establishes active or unlawful extraction.

Keep the existing Python and static MapLibre architecture. The final month does
not require an ILP solver, restoration-cost modelling, another application stack
or a new deep-learning model.

## Confirmed starting point

- The Railway site responds at https://peatland-web-production.up.railway.app;
  its dataset contains 82 sites.
- The local September build differs substantially from the deployed release.
  For Monivea 2022, the deployed GNG series contains 37.85 ha and the local
  pre-fix series 0.40 ha. These are different method versions.
- The existing 300 accuracy points have no reference labels. A working map,
  passing unit tests and NPWS plot counts are not a precision/recall assessment.
- The local pre-fix dataset has 506 missing expected overlay/clean JPEG assets.
  The September tile-build log was incomplete when inspected.

## Radiometry correction and controlled evidence

The previous reader assumed every spectral DN was reflectance multiplied by
10000. Sentinel-2 processing baseline 04.00 and later introduced an additive
offset. Monivea's 2022-05-15 product XML explicitly reports
`BOA_QUANTIFICATION_VALUE=10000` and `BOA_ADD_OFFSET=-1000` for the spectral bands.
Both baseline 04.00 and a reprocessed baseline 05.10 product exist for that day.
Use processing metadata, not acquisition year, because old imagery can be
reprocessed. [Copernicus documentation](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html).

On the frozen baseline 05.10 scene
`S2A_MSIL2A_20220515T114351_R123_T29UNV_20240616T041853`, keeping the same grid and
valid AOI, restoring the uncorrected +1000 DN reproduces **40.40 ha** under the
NDVI baseline; correcting it gives **0.75 ha**. With corrected inputs, spectral
rules, K-means and GNG each return **0.72 ha**, with zero pixel disagreement on
this particular scene. These are candidate-area measurements, not ground truth
or evidence that one classifier is more accurate.

The new reader harmonizes spectral channels, preserves SCL, excludes no-data and
honours explicit STAC scale/offset metadata. All multispectral methods now share
quality masks and contrast/burn rules. GNG normalization and training use only
finite, valid pixels inside the site. Prediction uses bounded pairwise-distance
blocks instead of a potentially >500 MB broadcast array. New runs record a
detector version and exact scene IDs; old caches cannot silently become new
results. Images are rendered from the saved scene IDs using the shared detector.

## Week 1: freeze reliable inputs and a pilot release

1. Run a complete one-site time series and image build into a new release folder.
   Keep the deployed dataset as an archived comparison, not an accuracy reference.
2. Check cloud/shadow masks, harmonization, grid alignment, tile-edge coverage and
   same-day reprocessed products on contrasting sites. The current per-window
   resampling is not yet a general cross-CRS reprojection/mosaic implementation.
3. Inspect raised and blanket bog controls after radiometry correction; reassess
   thresholds calibrated on uncorrected spectra. Freeze parameters before the
   held-out evaluation. The longitude-based bog-type tag remains a heuristic.
4. Freeze explicit scene IDs and a balanced seasonal selection for evaluation.
   The existing cached dates were chosen by GNG's seasonal maximum: they are
   useful controls, but not an independent sampling design for comparing methods.

## Week 2: independent labels and honest comparison

1. Agree the reference imagery source and dates with the supervisor. Annotate
   candidate bare peat, vegetation, water, burn, other and unsure. Record source,
   reference date and uncertainty. Do not copy detector predictions into truth.
2. Use separate calibration and held-out sites. Include water, burns, seasonally
   brown vegetation and small visible cut patches, not only documented hotspots.
3. Label the shuffled `labels.csv` without prediction/stratum columns. Score all
   four methods against the same reference points and frozen scenes.
4. Report precision, recall, F1, class confusions, runtime, and disagreements.
   Use recorded stratum population/sample sizes for weighted estimates. Claims
   apply to sampled site-years; broader inference needs a representative site
   design and uncertainty estimates. The scorer currently reports point estimates,
   not confidence intervals. [Olofsson et al., 2014](https://doi.org/10.1016/j.rse.2014.02.015).

## Week 3: final experiment and web release

1. Run seeds/node-budget/contrast ablations on the frozen evaluation set. Include
   the rules-only method to distinguish GNG's contribution from spectral guards.
2. Generate final regional results with versioned caches. Record evaluated scenes,
   missing years and effective valid coverage. A seasonal maximum is sensitive to
   the number and timing of observations; do not call it annual extraction area.
3. Produce all image pairs from the exact saved scene IDs. Require matching image
   manifests, finite areas, valid dates and dataset version before publication.
4. Review the full release on a local preview, then update Railway with the same
   data and assets. Keep the preceding release available for comparison.

## Week 4: finish the thesis and demonstration

Complete methods, results and limitations from the frozen experiment outputs;
record configs, versions, scene IDs and commands. Reserve several days for
supervisor corrections and demonstration rehearsal. If GNG matches a simpler
baseline, report that result and its computational trade-off; the system and
controlled comparison remain deliverables, subject to supervisor expectations.

## Commands

Run from the repository root. Use new paths for each experiment; commands below
do not overwrite the deployed dataset or the original annotation file.

```bash
python3 -m pytest tests/ -q
python3 scripts/12_method_comparison.py --case Monivea 2022 --skip-sensitivity --output-dir outputs/evaluation/control-v1
python3 scripts/12_method_comparison.py --from-run outputs/evaluation/control-v1 --output-dir outputs/evaluation/control-v1-repeat
python3 scripts/13_sample_points.py --run outputs/evaluation/control-v1 --output-dir outputs/evaluation/control-v1/annotation
# Fill the blind labels.csv from independent reference imagery, then:
python3 scripts/14_score_points.py outputs/evaluation/control-v1/annotation/accuracy_points.csv --labels outputs/evaluation/control-v1/annotation/labels.csv --output outputs/evaluation/control-v1/accuracy.json

python3 scripts/10_build_dataset.py --site Monivea --cache-dir outputs/cache-release-v1 --output outputs/releases/pilot/data/sites.geojson
python3 scripts/11_render_tiles.py --site Monivea --cache-dir outputs/cache-release-v1 --output-dir outputs/releases/pilot/data/tiles
python3 scripts/validate_dataset.py --dataset outputs/releases/pilot/data/sites.geojson --tiles outputs/releases/pilot/data/tiles --require-current-version
node --check web/app.js
```

The scorer exits **2** while labels are missing or unsure; undefined metrics are
`null`. Old CSVs without design weights yield descriptive sample metrics only.
The scene comparison writes the original scene arrays, prediction masks and
metadata so subsequent experiments and sampling can run without network access.
