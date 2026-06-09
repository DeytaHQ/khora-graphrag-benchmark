# khora-graphrag-benchmark — reproducible GraphRAG-Bench evaluation of Khora.
#
# Typical flow:
#   1. cp .env.example .env  &&  edit OPENAI_API_KEY
#   2. make setup            # install deps + start postgres/neo4j containers
#   3. make run-small        # ~10-15 min smoke test
#   4. make report           # regenerate JSON/MD/HTML reports from latest run
#
# Heavier runs:
#   make run-medium          # ~45-60 min, ~30% sampling
#   make run                 # ~2-3 h, full dataset

# Allow .env to set OPENAI_API_KEY, POSTGRES_URL, NEO4J_URL, KHORA_SPEC, etc.
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Prefer python3.13 when available; some upstream wheels (litellm, parts of
# the khora accel extras) don't ship cpython 3.14 wheels yet. Override with
# `PYTHON=python3 make setup` if you're on a different version.
PYTHON ?= $(shell command -v python3.13 || command -v python3)
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

KHORA_SPEC ?= khora[accel]==0.18.5

# Default DB URLs assume the docker-compose stack on localhost. Override in .env
# to point at your own Postgres / Neo4j (set the full URL, or just the *_PORT to
# match a remapped container port — both the URL and docker-compose honor it).
POSTGRES_PORT ?= 5432
NEO4J_BOLT_PORT ?= 7687
POSTGRES_URL ?= postgresql://bench:bench@localhost:$(POSTGRES_PORT)/khora_graphrag_bench
NEO4J_URL ?= bolt://neo4j:benchbench@localhost:$(NEO4J_BOLT_PORT)

export POSTGRES_URL
export NEO4J_URL
# Khora reads these names; mirror so users only set one.
export KHORA_DATABASE_URL = $(POSTGRES_URL)
export KHORA_NEO4J_URL = $(NEO4J_URL)

.DEFAULT_GOAL := help
.PHONY: help setup install docker-up docker-down docker-status reset-db \
        run run-small run-medium run-full report clean clean-all check-env check-python

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- Setup -------------------------------------------------------------------

setup: install docker-up  ## One-shot: create venv, install deps, start containers
	@echo ""
	@echo "✓ Setup complete. Next: make run-small"

check-python:
	@if [ -z "$(PYTHON)" ] || ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "✗ No suitable Python interpreter found (tried python3.13, python3)."; \
		echo "  Install Python 3.13. On Fedora: sudo dnf install python3.13"; \
		exit 1; \
	fi; \
	ver=$$($(PYTHON) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'); \
	if [ "$$ver" != "3.13" ]; then \
		echo "✗ Python 3.13 required, found $$ver via $(PYTHON)."; \
		echo "  The pinned khora + litellm stack does not support Python 3.14+ yet."; \
		echo "  Install Python 3.13 and re-run, e.g.:"; \
		echo "    sudo dnf install python3.13              # Fedora"; \
		echo "    brew install python@3.13                 # macOS"; \
		echo "    PYTHON=python3.13 make setup             # then point Makefile at it"; \
		exit 1; \
	fi; \
	echo "✓ Python 3.13 detected at $(PYTHON)"

$(VENV)/bin/activate: check-python
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel

install: $(VENV)/bin/activate  ## Install the benchmark package + pinned Khora
	@echo "→ Installing benchmark package + $(KHORA_SPEC)"
	$(PIP) install -e .
	@# Allow KHORA_SPEC override to replace the pinned khora spec.
	@DEFAULT_SPEC='khora[accel]==0.18.5'; \
	if [ "$(KHORA_SPEC)" != "$$DEFAULT_SPEC" ]; then \
		echo "→ Overriding khora install: $(KHORA_SPEC)"; \
		$(PIP) install --force-reinstall "$(KHORA_SPEC)"; \
	fi

docker-up:  ## Start Postgres + Neo4j containers
	docker compose up -d --wait
	@echo "✓ Postgres + Neo4j up"

docker-down:  ## Stop containers (keeps volumes)
	docker compose down

docker-status:  ## Show container health
	docker compose ps

reset-db:  ## Wipe Postgres + Neo4j volumes so the next run has zero stale data
	docker compose down -v
	docker compose up -d --wait
	@echo "✓ Fresh Postgres + Neo4j (volumes wiped)"

# --- Running the benchmark ---------------------------------------------------

check-env:
	@if [ -z "$$OPENAI_API_KEY" ] || [ "$$OPENAI_API_KEY" = "sk-..." ]; then \
		echo "✗ OPENAI_API_KEY is not set (or still the .env.example placeholder). Copy .env.example to .env and fill it in."; \
		exit 1; \
	fi

run: run-full  ## Alias for `make run-full`

run-small: check-env reset-db  ## Full pipeline, ~5% sampling (~10-15 min, smoke test)
	$(PY) -m khora_graphrag_bench.cli run --sample small

run-medium: check-env reset-db  ## Full pipeline, ~30% sampling (~45-60 min)
	$(PY) -m khora_graphrag_bench.cli run --sample medium

run-full: check-env reset-db  ## Full pipeline, 100% sampling (~2-3 h)
	$(PY) -m khora_graphrag_bench.cli run --sample full

# --- Reporting ---------------------------------------------------------------

report:  ## Regenerate JSON/MD/HTML reports from the latest run in results/
	$(PY) -m khora_graphrag_bench.cli report

# --- Cleanup -----------------------------------------------------------------

clean:  ## Remove venv + Python caches (keeps results/ and containers)
	rm -rf $(VENV) build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean docker-down  ## clean + stop containers + drop their volumes
	docker compose down -v
	rm -rf results/
