.PHONY: install shell fmt lint typecheck test run clean

install:
	poetry install

shell:
	poetry shell

fmt:
	poetry run black src tests
	poetry run isort src tests

lint:
	poetry run flake8 src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest --maxfail=1 --disable-warnings -q

run:
	poetry run risk-manager run  # calls your CLI or main()

clean:
	rm -rf .venv .pytest_cache __pycache__ build dist

.PHONY: check

check: fmt lint typecheck test run
	@echo "\n✅ All checks passed!"