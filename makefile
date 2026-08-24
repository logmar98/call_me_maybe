

all: run

install:
	@uv sync

run:
	@uv run python3 -m src

debug:
	@uv run python3 -m pdb -m src

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@rm -rf data/output

lint:
	@uv run flake8 src/
	@uv run mypy src/__main__.py --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs