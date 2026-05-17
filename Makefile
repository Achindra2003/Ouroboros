.PHONY: install dev test lint run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest ouroboros/tests/ -v

lint:
	ruff check ouroboros/ --fix

run:
	uvicorn ouroboros.server:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -f ouroboros.db ouroboros_sessions.db
