#!/bin/sh
# Execution receipt source: invoked one command at a time by the reviewer.
# Working directory: parent bp2004_acquisition_v2 directory.
# No --fetch command is present.

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/miniconda3/bin/python -m pytest -q -s -W error -p no:cacheprovider --basetemp validation_01/pytest_tmp test_collect_bp2004.py

PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python -m ruff check --no-cache --config ../../wavecert-fwi/pyproject.toml collect_bp2004.py test_collect_bp2004.py

# Initial full Ruff retains the existing v1 test-only I001 import-order finding.
# Keep all frozen candidate bytes; isolate collector and test E/F diagnostics.
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python -m ruff check --no-cache --config ../../wavecert-fwi/pyproject.toml collect_bp2004.py

PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python -m ruff check --no-cache --config ../../wavecert-fwi/pyproject.toml --select E,F test_collect_bp2004.py

# CLI --select caused the prior command to re-enable E501 despite repository
# ignore; repeat with the unchanged repository E501 ignore explicitly retained.
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python -m ruff check --no-cache --config ../../wavecert-fwi/pyproject.toml --select E,F --ignore E501 test_collect_bp2004.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/miniconda3/bin/python collect_bp2004.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/miniconda3/bin/python -W error validation_01/check_notice.py
