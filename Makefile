SHELL := /bin/bash
PYTHON := python3
POETRY := poetry
PKG := ghost
COMPOSE ?= docker compose

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
	$(PYTHON) cli.py --help

soc:
	$(PYTHON) -c "from app.soc import run_soc_main; import asyncio; asyncio.run(run_soc_main())"

up:
	$(COMPOSE) up --build

down:
	$(COMPOSE) down
