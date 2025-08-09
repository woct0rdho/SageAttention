# CI/CD Process for SageAttention

This document describes the continuous integration and deployment process for SageAttention.

## Overview

The CI/CD pipeline consists of a single unified workflow:

**CI - Build, Test, and Publish** (`.github/workflows/ci.yml`)
- Runs on both pull requests and pushes to main branch
- On PRs: Builds wheels for testing (no publishing)
- On main: Builds, tests, and publishes wheels to GitHub Packages

## Workflow Details

### CI - Build, Test, and Publish

**Trigger**: Pull requests and pushes to `main` branch

**Jobs**:
- **build-wheel**: Builds wheel using Docker
- **test-wheel**: Downloads and tests the built wheel (main branch only)
- **publish-wheel**: Publishes wheel to GitHub Packages (main branch only)

**Features**:
- **On PRs**: Builds wheel for testing (no publishing)
- **On main**: Builds, tests, and publishes wheel
- Proper wheel naming: `sageattention-2.2.0+cu129torch270-cp312-cp312-linux_x86_64.whl`
- Artifact retention for 30 days

## Wheel Naming Convention

Wheels are named according to the pattern:
```
sageattention-{version}+{cuda_version}torch{torch_version}-cp{python_version}-cp{python_version}-{platform}.whl
```

Example:
```
sageattention-2.2.0+cu129torch270-cp312-cp312-linux_x86_64.whl
```

Where:
- `2.2.0`: SageAttention version
- `cu129`: CUDA version (12.9)
- `torch270`: PyTorch version (2.7.0)
- `cp312`: Python version (3.12)
- `linux_x86_64`: Platform

## Configuration

### Environment Variables

The workflows use these environment variables:

- `CUDA_VERSION`: CUDA version (default: 12.9.1)
- `PYTHON_VERSION`: Python version (default: 3.12)
- `TORCH_MINOR_VERSION`: PyTorch minor version (default: 7)
- `TORCH_PATCH_VERSION`: PyTorch patch version (default: 0)
- `TORCH_CUDA_ARCH_LIST`: CUDA architectures to build for

These variables are used consistently across:
- `docker-bake.hcl` (as variables with defaults)
- CI workflows (as matrix values or explicit settings)
- `update_pyproject.py` (to set PyTorch version constraints)

### Secrets

The following secrets are required:

- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
- Repository permissions must allow package publishing

## GitHub Packages

### Publishing

Wheels are published to GitHub Packages using the PyPI registry:

```
https://github.com/{owner}/{repo}/packages/pypi
```

### Installation

Users can install from GitHub Packages:

```bash
pip install sageattention --index-url https://github.com/{owner}/{repo}/packages/pypi/
```

Or add to `requirements.txt`:
```
--index-url https://github.com/{owner}/{repo}/packages/pypi/
sageattention==2.2.0
```



## Docker Build Targets

The `docker-bake.hcl` provides these targets:

- **`wheel`**: Build wheel for distribution (default, outputs to `./wheels/`)
- **`dev`**: Development build for testing
- **`test`**: Test the built wheel

## Local Development

### Testing Workflows Locally

Use [act](https://github.com/nektos/act) to test workflows locally:

```bash
# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Test PR workflow
act pull_request

# Test build workflow
act push
```

### Manual Build

For manual builds, use the Docker build system:

```bash
# Build wheel (default)
docker buildx bake --file docker-bake.hcl

# Build wheel explicitly
docker buildx bake --file docker-bake.hcl wheel

# Development build
docker buildx bake --file docker-bake.hcl dev

# Test build
docker buildx bake --file docker-bake.hcl test
```

## Troubleshooting

### Common Issues

1. **Build failures**: Check CUDA version compatibility
2. **Publishing failures**: Verify repository permissions
3. **Test failures**: Ensure PyTorch version matches

### Debugging

1. Check workflow logs in GitHub Actions
2. Use `act` for local testing
3. Verify Docker build locally before pushing

### Performance

- Workflows use GitHub Actions cache for faster builds
- Multi-stage Docker builds reduce image size
- Parallel job execution where possible

## Security

- No secrets are exposed in logs
- Pull requests don't publish packages
- Only main branch triggers publishing
- Docker images are scanned for vulnerabilities

## Monitoring

- Workflow status is visible in GitHub repository
- Failed builds block merging to main
- Artifacts are retained for 30 days
- Docker images are tagged with commit SHAs 