# SIH26168 — named targets so nobody retypes flags (see docs/STEP2_TRAINING_PLAN.md).
# Local box: `smoke` only (stdlib). Colab T4: everything else. Studio: `apk`.
# Every python target needs PYTHONPATH=. (pytest reads pythonpath from pyproject).

PY := PYTHONPATH=. python

.PHONY: smoke test audit train train-smoke eval eval-2d harness export apk release-check

smoke: ## Local stdlib gate: compile + download skip-logic (runs anywhere)
	python3 -m compileall -q python tests
	python3 -c "import sys; sys.path.insert(0,'.'); \
	from pathlib import Path; import tempfile; \
	from python.download_iovnbd import download; \
	d = Path(tempfile.mkdtemp()); p = d/'s.zip'; p.write_bytes(b'x'*8); \
	download('http://example.invalid/s.zip', p, 8); print('smoke OK')"

test: ## Full pytest suite (Colab/CI: needs numpy pandas loguru)
	$(PY) -m pytest tests/ -q

audit: ## Per-file val audit, Step 0 (Colab T4, ~8 min)
	$(PY) python/eval_per_file.py --model experiments/checkpoints/model_avnet_stage1.p

train-smoke: ## 5-epoch stratified smoke (Colab T4, ~5 min)
	$(PY) python/train_avnet.py --epochs 5 --batch 256 --lr 1e-3 --device cuda \
		--augment-yaw --augment-bike --lambda-nll 0.1

train: ## Full 15-epoch run (Colab T4, ~15 min)
	$(PY) python/train_avnet.py --epochs 15 --batch 256 --lr 1e-3 --device cuda \
		--augment-yaw --augment-bike --lambda-nll 0.1

eval: ## Legacy 1D screening plot (byte-identical)
	$(PY) python/eval_drift.py --model experiments/checkpoints/model_avnet_stage1.p \
		--plot reports/drift_plot.png --mode 1d

eval-2d: ## Real 2D eval: ATE/RTE/drift (Colab)
	$(PY) python/eval_drift.py --model experiments/checkpoints/model_avnet_stage1.p \
		--plot reports/drift_2d.png --mode 2d

harness: ## InEKF replay on the stop-go segment (Colab)
	$(PY) python/inekf_harness.py --test-lean
	$(PY) python/inekf_harness.py --model experiments/checkpoints/model_avnet_stage1.p \
		--windows 600 --start 2000 --lean-mode car --q-acc 30.0

export: ## ONNX→TFLite + validation gate (Colab)
	$(PY) python/export_tflite.py --model experiments/checkpoints/model_avnet_stage1.p \
		--out model.tflite --onnx model.onnx

apk: ## Debug APK + unit tests (Studio/JDK17 box)
	gradle :app:assembleDebug :app:testDebugUnitTest --project-dir android

release-check: smoke ## Pre-tag gate: version + changelog sanity
	@grep -q "## \[Unreleased\]" CHANGELOG.md && echo "CHANGELOG OK"
	@test -n "$(git tag --points-at HEAD)" || echo "note: no tag on HEAD (cut one for releases)"
