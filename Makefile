.PHONY: all install dev-install test lint typecheck ci demo pipeline clean

PYTHON ?= python3
PIP ?= pip

# Default: show available targets
all:
	@echo "Benchmark Suite — available commands:"
	@echo "  make install      Install runtime dependencies"
	@echo "  make dev-install  Install dev + runtime dependencies (editable)"
	@echo "  make test         Run the test suite"
	@echo "  make lint         Run ruff"
	@echo "  make typecheck    Run mypy"
	@echo "  make ci           Run lint + typecheck + test"
	@echo "  make demo         Run the offline demo"
	@echo "  make pipeline     Run the full pipeline on sample data"
	@echo "  make clean        Remove build artifacts"

install:
	$(PIP) install -r requirements.txt

dev-install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy .

ci: lint typecheck test
	@echo "All CI checks passed."

demo:
	$(PYTHON) -m pipeline.demo --keep

pipeline:
	$(PYTHON) -m pipeline.demo

clean:
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
