PYTHONPATH := src
export PYTHONPATH
PY ?= .venv/bin/python

.PHONY: test data cache llm probe thresholds gate1 signals

test:
	$(PY) -m pytest

data:
	$(PY) scripts/01_build_data.py

cache:
	$(PY) scripts/02_cache_activations.py

llm:
	$(PY) scripts/03_score_llamaguard.py

probe:
	$(PY) scripts/04_train_probe.py

thresholds:
	$(PY) scripts/05_fit_thresholds.py

gate1:
	$(PY) scripts/06_gate1_counts.py
	$(PY) scripts/07_handcheck_sample.py

signals:
	$(PY) scripts/10_fit_signals.py
