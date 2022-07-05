.DEFAULT_GOAL := all
isort = isort pytest_speed tests
black = black pytest_speed tests

.PHONY: install
install:
	pip install -r tests/requirements.txt
	pip install -r tests/requirements-linting.txt
	poetry install
	pre-commit install

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	flake8 --max-complexity 12 --max-line-length 120 --ignore E203,W503 pytest_speed tests
	$(isort) --check-only --df
	$(black) --check

.PHONY: test
test:
	coverage run -m pytest

.PHONY: testcov
testcov: test
	@coverage report --show-missing
	@coverage html

.PHONY: mypy
mypy:
	mypy pytest_speed

.PHONY: all
all: lint mypy testcov
