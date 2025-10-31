.PHONY: venv
venv: ##@development Set up virtual environment
venv:
	${POETRY} install

.PHONY: build
buid: ##@development Build the docker images
build: prod_image ?= ${PROJECT}:${BRANCH_NAME}-${BUILD_NUMBER}
build: dev_image ?= ${PROJECT}_development:${BRANCH_NAME}-${BUILD_NUMBER}
build: args ?= -f docker/Dockerfile --build-arg PROJECT_DIR=. --network=host --build-arg BUILDKIT_INLINE_CACHE=1
build:
	DOCKER_BUILDKIT=1 docker build --progress=plain --target production -t ${prod_image} ${args} .
	DOCKER_BUILDKIT=1 docker build --progress=plain --target development -t ${dev_image} --cache-from ${prod_image} ${args} .

.PHONY: infrastructure
infrastructure: ##@development Set up infrastructure for tests
infrastructure:
	echo "Skipping infrastructure..."

.PHONY: clean
clean: ##@development Clean up any dependencies
clean:
	echo "Skipping clean..."

.PHONY: ci
ci: ##@development Run CI pipeline
ci: clean build infrastructure lint test clean