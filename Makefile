.PHONY: check test plan dashboard-build

check:
	python -m compileall -q src tests
	ruff check .
	ruff format --check .
	python -m pytest

test:
	python -m pytest

plan:
	enginebench plan --config configs/h100-1x.yaml --engine all

dashboard-build:
	npm --prefix dashboard run build
