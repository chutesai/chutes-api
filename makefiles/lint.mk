.PHONY: lint
lint: ##@lint Run all lint tools locally
lint:
	@echo "Running lint tools..."; \
	ruff check || exit 1; \
	ruff format --check --line-length 100 || exit 1; \

.PHONY: reformat
reformat: ##@local Reformat all packages or specific TARGET_PROJECT
reformat:
	@echo "Reformatting..."; \
	ruff format --line-length 100; \