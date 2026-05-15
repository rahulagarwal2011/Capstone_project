.PHONY: smoke test lint format bench docker-up docker-down docker-build docker-shell docker-smoke docker-test docker-lint docker-logs k8s clean install dev

install:
	uv pip install -e ".[dev]"

dev:
	uv pip install -e ".[dev,train,eval]"

smoke:
	PYTHONPATH=src python scripts/smoke.py

test:
	PYTHONPATH=src pytest tests/unit -x --cov=reason_reduce --cov-report=term-missing

test-all:
	PYTHONPATH=src pytest tests/ --cov=reason_reduce --cov-report=html

test-integration:
	PYTHONPATH=src pytest tests/integration -x

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

bench:
	python benchmarks/run_benchmark.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

docker-build:
	docker compose build app

docker-shell:
	docker compose exec app bash

docker-smoke:
	docker compose exec app python scripts/smoke.py

docker-test:
	docker compose exec app pytest tests/ -q

docker-lint:
	docker compose exec app ruff check src/ tests/

docker-logs:
	docker compose logs -f app ray-head

# === Phase Validation ===
phase1-validate:
	PYTHONPATH=src python scripts/validate_phase1.py

docker-phase1:
	docker compose exec app python scripts/validate_phase1.py

k8s-local:
	kind create cluster --name reason-reduce
	kubectl apply -k deploy/k8s/

k8s-teardown:
	kind delete cluster --name reason-reduce

reproduce:
	bash scripts/reproduce_results.sh

clean:
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
