# Contributing to SageAttention (Fork)

This guide details the enhancements made in this fork, specifically regarding **Docker-based builds**, **CI/CD standards** and **Pre-built wheels**.

## 🚀 Quick Start for Developers

### Using Docker (Recommended)
We use `docker buildx bake` to manage complex multi-platform build configurations. This ensures reproducible environments for both Linux and Windows (cross-compilation) artifacts.

```bash
# Build default configuration (Linux + PyTorch 2.8 + CUDA 12.9)
docker buildx bake default

# Build all Linux wheels
docker buildx bake linux
```

**Note on Windows**: Windows wheels are currently built using **native GitHub Actions runners** (or locally via `pip`) rather than Docker, as cross-compilation for CUDA kernels is complex. See the "Windows Support" section below.

### Local Testing
You can run the CI workflows locally using [`act`](https://nektosact.com/installation/index.html) (requires Docker):
```bash
# Run the Linux wheel build workflow
gh act -W .github/workflows/build-wheels-linux.yml --container-architecture linux/amd64
```

## 📦 CI/CD & Standards

### Wheel Naming Convention
We enforce specific build tags to ensure wheels are strictly PEP 440 compliant while carrying dependency metadata:
*   **Format**: `sageattention-{version}-{build_tag}-...`
*   **Example**: `sageattention-2.2.0-8.9-cp312-cp312-linux_x86_64.whl`
    *   `8.9` represents **PyTorch 2.8** + **CUDA 12.9**.
    *   This prevents `InvalidVersion` errors during installation.

### Workflows
*   **`build-wheels-linux.yml`**: Uses `setup-python` and native runners for speed.
*   **`build-wheels-windows.yml`**: Uses MSVC runners.
*   Both workflows verify the generated wheels by installing them and running a basic import check.

## 📦 Pre-built Wheels
We provide pre-built wheels for the following configurations:

| PyTorch | CUDA | Python | Platform |
| :---: | :---: | :---: | :---: |
| 2.5.1 | 12.4 | 3.12 | Linux, Windows |
| 2.6.0 | 12.6 | 3.12 | Linux, Windows |

*For other configurations, please build from source.*