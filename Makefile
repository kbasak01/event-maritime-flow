PY ?= python
.PHONY: lint test test-data smoke train-smoke reproduce-tables
lint:            ; ruff check src tests scripts
test:            ; pytest -q
test-data:       ; pytest -q -m "dataset" -p no:cacheprovider
smoke:           ; $(PY) -c "import evflow; print('ok')" && pytest -q tests/test_smoke.py
train-smoke:     ; $(PY) -m evflow.train.loop --config configs/train_real_sup.yaml --smoke
reproduce-tables:; $(PY) scripts/make_tables.py --check
