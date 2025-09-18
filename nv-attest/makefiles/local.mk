.PHONY: bandit-local
bandit-local: ##@lint Run bandit
bandit-local: files ?= ${SERVICE}
bandit-local:
	${POETRY} run bandit -r ${files}

.PHONY: black-local
black-local: ##@lint Run black
black-local: files ?= ${SERVICE} tests
black-local:
	${POETRY} run black ${files}

.PHONY: flake8-local
flake8-local: ##@lint Run flake8
flake8-local: files ?= ${SERVICE} tests
flake8-local:
	${POETRY} run flake8 --config .flake8 ${files}

.PHONY: isort-local
isort-local: ##@lint Run isort
isort-local: files ?= ${SERVICE} tests
isort-local: args ?= --diff --check-only --quiet -rc ${files}
isort-local:
	${POETRY} run isort ${args}

.PHONY: mypy-local
mypy-local: ##@lint Run mypy
mypy-local: args ?= -p ${SERVICE}
mypy-local:
	${POETRY} run mypy ${args}

.PHONY: lint-local
lint-local: ##@lint Run lint tools
lint-local: bandit-local black-local flake8-local isort-local mypy-local

.PHONY: clean-imports
clean-imports: ##@local Remove unused imports
clean-imports: 
	autoflake --in-place --remove-all-unused-imports --recursive ${SERVICE} tests

.PHONY: reformat
reformat: ##@local Reformat module
reformat: files ?= ${SERVICE} tests
reformat: clean-imports
	${POETRY} run isort --overwrite-in-place ${files}
	${POETRY} run black ${files}

PHONY: test-local
test-local: ##@local Run test suite
test-local: venv
	${POETRY} run pytest -s --tb=native --durations=5 --cov=${SERVICE} --cov-report=html tests
	${POETRY} run coverage report --fail-under=90