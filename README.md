# [SageAttention](https://github.com/thu-ml/SageAttention) Fork - Build System Integration

This repository is a fork of the original SageAttention project that provides enhanced build system integration and automated CI/CD pipeline for distributing pre-built Python wheels across multiple Python, PyTorch, and CUDA versions.

## What is SageAttention?

SageAttention is an efficient and accurate attention mechanism that uses **low-bit quantization** to significantly reduce memory usage and increase speed while maintaining accuracy. It provides:

- **INT8 quantization** for Query and Key tensors
- **FP8/FP16 quantization** for Value tensors  
- **Hardware-optimized kernels** for different GPU architectures (SM80, SM89, SM90)
- **Plug-and-play integration** - easily replace `scaled_dot_product_attention` in existing models
- **Support for multiple tensor layouts** (HND, NHD)
- **Variable-length sequence support**

## What This Fork Adds

This fork extends the original SageAttention project with:

1. **Automated CI/CD Pipeline** - Builds wheels for multiple configurations automatically
2. **GitHub Packages Distribution** - Pre-built wheels available for easy installation
3. **Docker-based Build System** - Consistent builds across different environments
4. **Multi-version Support** - Wheels built for different Python/PyTorch/CUDA combinations

## CI/CD Pipeline

Our GitHub Actions CI pipeline automatically:

1. **Builds wheels** for the following configuration:
   - Python: 3.12
   - PyTorch: 2.7.0
   - CUDA: 12.9.1
   - Supported GPU architectures: 7.0, 7.5, 8.0, 8.6, 8.9, 9.0

2. **Tests the wheels** using Docker containers to ensure compatibility

3. **Publishes to GitHub Packages** with proper naming convention:
   ```
   sageattention-{version}+cu{minor}9torch{major}{patch}-cp{python_version}-linux_x86_64.whl
   ```

4. **Triggers on** pushes to main branch (not on pull requests)

## Available Packages

Once the CI pipeline completes successfully, pre-built wheels are available in the [GitHub Packages](https://github.com/pixeloven/SageAttention/packages) section of this repository.

## Building Locally

### Prerequisites

- CUDA 12.0 or higher
- Python 3.9+
- PyTorch with CUDA support
- Compatible GPU (compute capability 8.0+)

### Method 1: Direct Installation

```bash
# Clone the repository
git clone https://github.com/pixeloven/SageAttention.git
cd SageAttention

# Install dependencies (adjust torch version as needed)
pip install torch==2.7.0+cu129 --index-url https://download.pytorch.org/whl/cu129

# Install SageAttention
python setup.py install
```

### Method 2: Docker Build

```bash
# Build using Docker (requires Docker Buildx)
docker buildx bake --file docker-bake.hcl wheel

# The wheel will be available in the ./wheels directory
```

### Method 3: Development Installation

```bash
# For development with editable installation
pip install -e .
```

## Using in Downstream Projects

### Installation from GitHub Packages

```bash
# Install from GitHub Packages
pip install sageattention --index-url https://github.com/pixeloven/SageAttention/packages/pypi/simple/
```
