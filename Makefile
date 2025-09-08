# SageAttention Build and Validation Makefile

.PHONY: validate-dockerfiles build-linux test-dockerfiles help

# Default target
help:
	@echo "SageAttention Build Commands:"
	@echo "  validate-dockerfiles  - Validate Dockerfiles with hadolint"
	@echo "  build-linux          - Build Linux wheels"
	@echo "  build-windows        - Build Windows wheels"
	@echo "  test-dockerfiles     - Test Dockerfile syntax quickly"
	@echo "  help                 - Show this help message"

# Validate Dockerfiles using hadolint
validate-dockerfiles:
	@echo "🔍 Validating Dockerfiles..."
	@docker pull hadolint/hadolint:latest-debian
	@echo "Validating dockerfile.builder.linux..."
	@docker run --rm -i -v $(PWD)/.hadolint.yaml:/.hadolint.yaml hadolint/hadolint:latest-debian hadolint --config /.hadolint.yaml - < dockerfile.builder.linux
	@echo "Validating dockerfile.builder.windows..."
	@docker run --rm -i -v $(PWD)/.hadolint.yaml:/.hadolint.yaml hadolint/hadolint:latest-debian hadolint --config /.hadolint.yaml - < dockerfile.builder.windows
	@echo "✅ All Dockerfiles validated successfully!"

# Quick syntax test (faster than full hadolint)
test-dockerfiles:
	@echo "🔍 Quick Dockerfile syntax test..."
	@docker build --target runtime -f dockerfile.builder.linux --dry-run . > /dev/null && echo "✅ Linux Dockerfile syntax OK"
	@echo "Note: Windows Dockerfile syntax test requires Windows Docker host"

# Build Linux wheels
build-linux:
	@echo "🔨 Building Linux wheels..."
	docker buildx bake --file docker-bake.hcl linux
