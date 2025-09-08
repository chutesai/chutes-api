.PHONY: infrastructure
infrastructure: ##@development Set up infrastructure for tests
infrastructure:
	${DC} up -d
	./tests/scripts/setup-cosign.sh

.PHONY: clean
clean: ##@development Clean up any dependencies
clean:
	${DC} down --remove-orphans --volumes