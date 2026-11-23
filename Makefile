.PHONY: install test lint format demo train clean

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest --cov=customer_sat --cov-report=term-missing

lint:
	ruff check src tests

format:
	ruff check --fix src tests && ruff format src tests

demo:            ## train on synthetic data, no dataset required
	customer-sat demo

train:           ## train on your data: make train DATA=data/application_train.csv
	customer-sat train --config configs/default.yaml --data $(DATA)

clean:
	rm -rf artifacts mlruns .pytest_cache .ruff_cache .coverage build dist *.egg-info
