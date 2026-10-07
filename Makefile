# Offline reproduction of the evaluation from the frozen runs in outputs/.
# Every target below reads frozen scene arrays and labels; none needs network
# access. `make all` reruns the tests and rebuilds every accuracy table.

PY      ?= python3
SAMPLE  := outputs/evaluation/2026-09-13-v3-sample
TEST    := outputs/evaluation/2026-10-07-v4-test
METHODS := ndvi rules gng v4_rules v4_kmeans v4_gng
V5TEST  := outputs/evaluation/2026-10-08-v5-test
V5EXT   := outputs/evaluation/2026-10-08-v5-external

.PHONY: all test v4 accuracy accuracy-sample accuracy-test accuracy-v5 detection-limit

all: test v4 accuracy accuracy-v5 detection-limit

test:
	$(PY) -m pytest -q tests

# v4 post-filter predictions on both frozen runs
v4:
	$(PY) scripts/29_v4_predictions.py $(SAMPLE)
	$(PY) scripts/29_v4_predictions.py $(TEST)

accuracy: accuracy-sample accuracy-test

accuracy-sample:
	$(PY) scripts/28_accuracy_report.py $(SAMPLE)
	$(PY) scripts/28_accuracy_report.py $(SAMPLE) --points $(SAMPLE)/annotation/accuracy_points_v4.csv \
		--methods $(METHODS) --tag _v4

accuracy-test:
	$(PY) scripts/28_accuracy_report.py $(TEST) --points $(TEST)/annotation/accuracy_points_v4.csv \
		--methods $(METHODS) --tag _v4

# detector v5 (SWIR peat) on its independent and external tests
accuracy-v5:
	$(PY) scripts/28_accuracy_report.py $(V5TEST) --methods ndvi rules kmeans gng v5 --tag _v5
	$(PY) scripts/28_accuracy_report.py $(V5EXT) --methods ndvi rules kmeans gng v5 --tag _v5

# strip-width detection limit (offline, from frozen scenes)
detection-limit:
	$(PY) scripts/33_detection_limit.py
