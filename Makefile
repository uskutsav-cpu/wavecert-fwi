.PHONY: install test lint demo clean

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check src tests

demo:
	wavecert demo --output results/demo

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
