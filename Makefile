SHELL := /bin/bash
PYTHON := python3
POETRY := poetry
PKG := ghost

.PHONY: venv install lint type test cov fmt up down api cli soc

venv:
	$(PYTHON) -m venv .venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

fmt:
	ruff check . --fix
	ruff format .
	$(PYTHON) -m black . || true

lint:
	ruff check .
	mypy app

type: lint

test:
	pytest -q

cov:
	pytest --cov=app --cov-report=term-missing

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

cli:
	python cli.py --help

soc:
	python -c "from app.soc import run_soc_main; import asyncio; asyncio.run(run_soc_main())"

up:
	docker-compose up --build

down:
	docker-compose down