# SageAttention Docker Bake Configuration - Optimized Multi-Stage Builds
# Clear naming convention: {platform}-{pytorch}-{cuda}-{python}
# Uses optimized multi-stage architecture with shared dependency layers
# Generates PEP 427 compliant wheels with build tags: sageattention-2.2.0-{torch_ver}.{cuda_ver}-cp312-cp312-{platform}.whl

# Build configuration variables
variable "PYTHON_VERSION" {
  default = "3.12"
}

variable "CUDA_VERSION" {
  default = "12.9.1"
}

variable "TORCH_VERSION" {
  default = "2.8.0"
}

variable "TORCH_CUDA_ARCH_LIST" {
  default = "8.0;8.6;8.9;9.0;12.0"
}

# Linux Builds
# Format: linux-pytorch{cuda}-{python}
target "linux-pytorch27-cu128-python312" {
  dockerfile = "dockerfile.builder.linux"
  target = "wheel"
  platforms = ["linux/amd64"]
  output = ["type=local,dest=./dist"]
  args = {
    CUDA_VERSION = "12.8.1"
    PYTHON_VERSION = PYTHON_VERSION
    TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST
    TORCH_VERSION = "2.7.0"
  }
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
}

target "linux-pytorch28-cu129-python312" {
  dockerfile = "dockerfile.builder.linux"
  target = "wheel"
  platforms = ["linux/amd64"]
  output = ["type=local,dest=./dist"]
  args = {
    CUDA_VERSION = "12.9.1"
    PYTHON_VERSION = PYTHON_VERSION
    TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST
    TORCH_VERSION = "2.8.0"
  }
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
}


# Test Targets
# Format: test-{platform}-pytorch{cuda}-{python}
target "test-linux-pytorch27-cu128-python312" {
  dockerfile = "dockerfile.builder.linux"
  target = "test"
  platforms = ["linux/amd64"]
  args = {
    CUDA_VERSION = "12.8.1"
    PYTHON_VERSION = PYTHON_VERSION
    TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST
    TORCH_VERSION = "2.7.0"
  }
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
}

target "test-linux-pytorch28-cu129-python312" {
  dockerfile = "dockerfile.builder.linux"
  target = "test"
  platforms = ["linux/amd64"]
  args = {
    CUDA_VERSION = "12.9.1"
    PYTHON_VERSION = PYTHON_VERSION
    TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST
    TORCH_VERSION = "2.8.0"
  }
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
}


# Convenience Groups
group "default" {
  targets = ["linux-pytorch28-cu129-python312"]
}

group "linux" {
  targets = [
    "linux-pytorch27-cu128-python312",
    "linux-pytorch28-cu129-python312"
  ]
}


group "pytorch27" {
  targets = [
    "linux-pytorch27-cu128-python312"
  ]
}

group "pytorch28" {
  targets = [
    "linux-pytorch28-cu129-python312"
  ]
}

group "all" {
  targets = [
    "linux-pytorch27-cu128-python312",
    "linux-pytorch28-cu129-python312"
  ]
}

group "test-all" {
  targets = [
    "test-linux-pytorch27-cu128-python312",
    "test-linux-pytorch28-cu129-python312"
  ]
} 