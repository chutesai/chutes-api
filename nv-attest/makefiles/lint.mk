.PHONY: bandit
bandit: ##@lint Run bandit
bandit: files ?= ${SERVICE}
bandit:
	${DC} run --rm --no-deps bandit -r ${files}

.PHONY: black
black: ##@lint Run black
black: files ?= ${SERVICE} tests
black:
	${DC} run --rm --no-deps black ${files}

.PHONY: flake8
flake8: ##@lint Run flake8
flake8: files ?= ${SERVICE} tests
flake8:
	${DC} run --rm --no-deps flake8 --config .flake8 ${files}

.PHONY: isort
isort: ##@lint Run isort
isort: files ?= ${SERVICE} tests
isort: args ?= --diff --check-only --quiet -rc ${files}
isort:
	${DC} run --rm --no-deps isort ${args}

.PHONY: mypy
mypy: ##@lint Run mypy
mypy: args ?= -p ${SERVICE}
mypy:
	${DC} run --rm --no-deps mypy ${args}

.PHONY: lint
lint: ##@lint Run lint tools
lint: bandit black flake8 isort mypy