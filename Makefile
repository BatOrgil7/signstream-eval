.PHONY: setup lint type test smoke figures reproduce

setup:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

type:
	mypy src/signstream

test:
	pytest

# The targets below come online in later increments, once the runner,
# simulator, and figure engine exist.
smoke:
	@echo "make smoke: not implemented yet (requires the runner and the tinyset pipeline)."
	@exit 1

figures:
	@echo "make figures: not implemented yet (requires metrics and the figure engine)."
	@exit 1

reproduce:
	@echo "make reproduce: not implemented yet (requires released emission logs and the score/stats/figures stages)."
	@exit 1
