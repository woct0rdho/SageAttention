# SageAttention Docker Build Guide

This guide explains how to build SageAttention wheels using Docker, which provides a consistent build environment across different systems.

## Prerequisites

- Docker installed and running
- Docker Buildx (recommended for advanced features)

## Quick Start

### Using Docker Buildx with docker-bake.hcl (Recommended)

The easiest way to build SageAttention is using Docker Buildx with the `docker-bake.hcl` file:

```bash
# Build wheel (default)
docker buildx bake --file docker-bake.hcl wheel

# Build for development
docker buildx bake --file docker-bake.hcl dev

# Build and test
docker buildx bake --file docker-bake.hcl full

# Build all targets
docker buildx bake --file docker-bake.hcl
```

### Using Direct Docker Commands

If you prefer to use regular Docker build commands:

```bash
# Build wheel
docker build -f dockerfile.builder --target sageattention-wheel --output type=local,dest=./wheels .

# Build development image
docker build -f dockerfile.builder --target sageattention-builder -t sageattention:dev .

# Build and test
docker build -f dockerfile.builder --target sageattention-test -t sageattention:latest .
```

## Build Targets

The Dockerfile provides several build targets:

- **`sageattention-builder`**: Builds the SageAttention package (development target)
- **`sageattention-wheel`**: Extracts the built wheel to local filesystem
- **`sageattention-test`**: Builds and tests the wheel in a runtime environment

The `docker-bake.hcl` provides additional convenience targets:

- **`dev`**: Development image with build tools
- **`wheel`**: Production wheel build (outputs to `./dist/`)
- **`full`**: Complete build with testing

## Configuration

### Environment Variables

You can customize the build by setting these variables in `docker-bake.hcl` or passing them as build arguments:

- `CUDA_VERSION`: CUDA version to use (default: 12.9.1)
- `PYTHON_VERSION`: Python version to use (default: 3.12)
- `TORCH_CUDA_ARCH_LIST`: CUDA architectures to build for (default: 7.0;7.5;8.0;8.6;8.9;9.0)
- `BUILD_PLATFORM`: Target platform (default: linux/amd64)

### Customizing Build Parameters

#### Using docker-bake.hcl variables

```bash
# Override variables when building
docker buildx bake --file docker-bake.hcl --set "*.args.CUDA_VERSION=12.8.0" wheel

# Build for different platform
docker buildx bake --file docker-bake.hcl --set "*.platforms=linux/arm64" wheel

# Build with custom architecture list
docker buildx bake --file docker-bake.hcl --set "*.args.TORCH_CUDA_ARCH_LIST=8.0;8.6;9.0" wheel
```

#### Using direct Docker commands

```bash
# Build with custom build arguments
docker build -f dockerfile.builder --target sageattention-wheel \
  --build-arg CUDA_VERSION=12.8.0 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" \
  --output type=local,dest=./wheels .
```

### CUDA Architecture List

The default architecture list includes common GPU architectures:
- `7.0`: Tesla V100, Titan V
- `7.5`: RTX 2080, GTX 1080 Ti
- `8.0`: A100, RTX 3090, RTX 4090
- `8.6`: RTX 3080, RTX 4080
- `8.9`: H100, L40
- `9.0`: H200, B200

You can customize this based on your target hardware:

```bash
# Build only for specific architectures
docker buildx bake --file docker-bake.hcl --set "*.args.TORCH_CUDA_ARCH_LIST=8.0;8.6;9.0" wheel

# Build for all modern architectures
docker buildx bake --file docker-bake.hcl --set "*.args.TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0" wheel
```

## Advanced Usage

### Multi-platform Builds

```bash
# Build for multiple platforms
docker buildx bake --file docker-bake.hcl --set "*.platforms=linux/amd64,linux/arm64" wheel
```

### Build Caching

The `docker-bake.hcl` includes GitHub Actions cache configuration:

```bash
# Use GitHub Actions cache (if available)
docker buildx bake --file docker-bake.hcl wheel

# Disable caching
docker buildx bake --file docker-bake.hcl --no-cache wheel
```

### Parallel Builds

```bash
# Build multiple targets in parallel
docker buildx bake --file docker-bake.hcl dev wheel full
```

## Output Locations

- **Wheels**: Built wheels are extracted to `./wheels/` or `./dist/` depending on the target
- **Images**: Docker images are tagged as `sageattention:dev`, `sageattention:wheel`, or `sageattention:latest`

## Troubleshooting

### Build Failures

1. **CUDA version mismatch**: Ensure your CUDA version is compatible with PyTorch
2. **Memory issues**: Increase Docker memory limit if building fails due to OOM
3. **Platform issues**: Use `--platform` flag to specify target architecture

### Performance Optimization

1. **Use Buildx caching**: Docker Buildx provides better caching for faster rebuilds
2. **Parallel builds**: Build multiple targets simultaneously
3. **Multi-stage builds**: The Dockerfile uses multi-stage builds to reduce final image size

### Common Issues

- **Docker not found**: Ensure Docker is installed and running
- **Buildx not available**: Use regular `docker build` commands as fallback
- **Permission issues**: Ensure you have proper permissions to run Docker commands

## Examples

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Build SageAttention wheel
  run: |
    docker buildx bake --file docker-bake.hcl --set "*.args.TORCH_CUDA_ARCH_LIST=8.0;8.6;9.0" wheel
  
- name: Upload wheel artifact
  uses: actions/upload-artifact@v3
  with:
    name: sageattention-wheel
    path: dist/
```

### Custom Build Environment

```bash
# Build with custom Python version
docker buildx bake --file docker-bake.hcl --set "*.args.PYTHON_VERSION=3.11" wheel

# Build for specific CUDA version
docker buildx bake --file docker-bake.hcl --set "*.args.CUDA_VERSION=12.8.0" wheel

# Build for ARM64 platform
docker buildx bake --file docker-bake.hcl --set "*.platforms=linux/arm64" wheel
```

### Development Workflow

```bash
# Build development environment
docker buildx bake --file docker-bake.hcl dev

# Run development container
docker run -it --gpus all sageattention:dev bash

# Build wheel from development container
docker run --rm -v $(pwd):/workspace sageattention:dev bash -c "cd /workspace && python setup.py bdist_wheel"
```

## File Structure

```
.
├── dockerfile.builder      # Main Dockerfile for building
├── docker-bake.hcl        # Docker Buildx bake configuration
├── .dockerignore          # Files to exclude from build context
├── wheels/                # Output directory for wheels (created automatically)
└── dist/                  # Alternative output directory (created automatically)
```

## Command Reference

### Docker Buildx Bake Commands

```bash
# Basic build commands
docker buildx bake --file docker-bake.hcl [target]

# Available targets
docker buildx bake --file docker-bake.hcl dev      # Development image
docker buildx bake --file docker-bake.hcl wheel    # Build wheel
docker buildx bake --file docker-bake.hcl full     # Build and test
docker buildx bake --file docker-bake.hcl          # Build all targets

# Customization
docker buildx bake --file docker-bake.hcl --set "*.args.VARIABLE=value" [target]
docker buildx bake --file docker-bake.hcl --set "*.platforms=platform" [target]
docker buildx bake --file docker-bake.hcl --no-cache [target]
```

### Direct Docker Build Commands

```bash
# Basic build commands
docker build -f dockerfile.builder --target [target] [options] .

# Available targets
docker build -f dockerfile.builder --target sageattention-builder -t sageattention:dev .
docker build -f dockerfile.builder --target sageattention-wheel --output type=local,dest=./wheels .
docker build -f dockerfile.builder --target sageattention-test -t sageattention:latest .

# With custom build arguments
docker build -f dockerfile.builder --target sageattention-wheel \
  --build-arg CUDA_VERSION=12.8.0 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" \
  --output type=local,dest=./wheels .
``` 