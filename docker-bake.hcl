# SageAttention Docker Bake Configuration - Comprehensive Build Matrix
# Supports SageAttention v2.2.0 (all GPUs) and SageAttention3 v1.0.0 (Blackwell only)
#
# Naming convention: {platform}-{package}-{pytorch}-{cuda}-{python}
#   - platform: linux
#   - package: sage2 (SageAttention 2/2++) or sage3 (SageAttention3)
#   - pytorch: pytorch26, pytorch27, pytorch28
#   - cuda: cu126, cu128, cu129
#   - python: python39, python310, python311, python312, python313
#
# Wheel naming (PEP 427):
#   sageattention-2.2.0-{torch}.{cuda}-cp{py}-cp{py}-linux_x86_64.whl
#   sageattn3-1.0.0-{torch}.{cuda}-cp{py}-cp{py}-linux_x86_64.whl

# ============================================================================
# Build Configuration Variables
# ============================================================================

variable "PYTHON_VERSION" {
  default = "3.12"
}

variable "CUDA_VERSION" {
  default = "12.9.1"
}

variable "TORCH_VERSION" {
  default = "2.8.0"
}

# CUDA 12.8-safe arch list (drops sm121 which isn't supported by nvcc 12.8)
variable "TORCH_CUDA_ARCH_LIST_CU128" {
  default = "7.0;7.5;8.0;8.6;8.7;8.9;9.0;10.0;12.0"
}

# CUDA 12.9+ arch list (full coverage including sm121)
variable "TORCH_CUDA_ARCH_LIST_CU129" {
  default = "7.0;7.5;8.0;8.6;8.7;8.9;9.0;10.0;12.0;12.1"
}

# Blackwell-only arch list for SageAttention3
variable "TORCH_CUDA_ARCH_LIST_BLACKWELL" {
  default = "10.0;12.0"
}

# ============================================================================
# SageAttention 2/2++ Builds (Main Package) - aligned to CI matrix
# ============================================================================

# CI builds two wheels: PyTorch 2.7 + CUDA 12.8 (Py3.12) and PyTorch 2.8 + CUDA 12.9 (Py3.12).

# ---------- PyTorch 2.5.1 + CUDA 12.4 ----------

target "linux-sage2-pytorch25-cu124-python312" {
  dockerfile = "dockerfile.builder.linux"
  target = "wheel"
  platforms = ["linux/amd64"]
  output = ["type=local,dest=./dist"]
  args = {
    CUDA_VERSION = "12.4.1"
    PYTHON_VERSION = "3.12"
    TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST_CU128
    TORCH_VERSION = "2.5.1"
    MAX_JOBS = "4"
  }
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
}

# ---------- PyTorch 2.6.0 + CUDA 12.6 ----------

target "linux-sage2-pytorch26-cu128-python312" {
  dockerfile = "dockerfile.builder.linux"
  target = "wheel"
  platforms = ["linux/amd64"]
  output = ["type=local,dest=./dist"]
  args = {
    CUDA_VERSION = "12.8.1"
    PYTHON_VERSION = "3.12"
    TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST_CU129
    TORCH_VERSION = "2.6.0"
    MAX_JOBS = "4"
  }
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
}

# ============================================================================
# SageAttention3 Builds (Blackwell-only, requires CUDA 12.8+, PyTorch 2.8+)
# ============================================================================
# Note: These would require a separate Dockerfile that builds from
# sageattention3_blackwell/ directory. Currently not implemented.

# target "linux-sage3-pytorch28-cu128-python310" {
#   dockerfile = "dockerfile.builder.sage3.linux"
#   target = "wheel"
#   platforms = ["linux/amd64"]
#   output = ["type=local,dest=./dist"]
#   args = {
#     CUDA_VERSION = "12.8.1"
#     PYTHON_VERSION = "3.10"
#     TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST_BLACKWELL
#     TORCH_VERSION = "2.8.0"
#   }
#   cache-from = ["type=gha"]
#   cache-to = ["type=gha,mode=max"]
# }

# ============================================================================
# Test Targets
# ============================================================================

target "test-linux-sage2-pytorch25-cu124-python312" {
  dockerfile = "dockerfile.builder.linux"
  target = "test"
  platforms = ["linux/amd64"]
  args = {
    CUDA_VERSION = "12.4.1"
    PYTHON_VERSION = "3.12"
    TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST_CU128
    TORCH_VERSION = "2.5.1"
  }
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
}

target "test-linux-sage2-pytorch26-cu128-python312" {
  dockerfile = "dockerfile.builder.linux"
  target = "test"
  platforms = ["linux/amd64"]
  args = {
    CUDA_VERSION = "12.8.1"
    PYTHON_VERSION = "3.12"
    TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST_CU129
    TORCH_VERSION = "2.6.0"
  }
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
}

# ============================================================================
# Convenience Groups
# ============================================================================

group "default" {
  targets = ["linux-sage2-pytorch26-cu128-python312"]
}

# CI-aligned builds (both wheels)
group "sage2-all" {
  targets = [
    "linux-sage2-pytorch25-cu124-python312",
    "linux-sage2-pytorch26-cu128-python312",
  ]
}

# Latest stable combinations (same as CI)
group "stable" {
  targets = [
    "linux-sage2-pytorch25-cu124-python312",
    "linux-sage2-pytorch26-cu128-python312",
  ]
}

# All tests
group "test-all" {
  targets = [
    "test-linux-sage2-pytorch25-cu124-python312",
    "test-linux-sage2-pytorch26-cu128-python312",
  ]
}

# Legacy aliases (backward compatibility)
group "linux" {
  targets = [
    "linux-sage2-pytorch25-cu124-python312",
    "linux-sage2-pytorch26-cu128-python312",
  ]
}

group "all" {
  targets = [
    "linux-sage2-pytorch25-cu124-python312",
    "linux-sage2-pytorch26-cu128-python312",
  ]
}
