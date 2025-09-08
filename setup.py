"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
import threading
from packaging.version import parse, Version
import warnings

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from wheel.bdist_wheel import bdist_wheel

# Compiler flags - detect MSVC compiler rather than OS
def is_msvc_compiler():
    """Detect if we're using MSVC compiler."""
    try:
        # Try to import distutils compiler to detect MSVC
        from distutils.msvccompiler import MSVCCompiler
        from distutils.util import get_platform
        from setuptools._distutils.msvccompiler import MSVCCompiler as SetuptoolsMSVCCompiler
        
        # Check if we're on Windows and likely using MSVC
        if os.name == "nt":
            # Additional check: look for cl.exe in PATH or VS environment
            import shutil
            return shutil.which("cl.exe") is not None or os.environ.get("VCINSTALLDIR") is not None
        return False
    except ImportError:
        # Fallback to OS detection if distutils unavailable
        return os.name == "nt"

if is_msvc_compiler():
    CXX_FLAGS = ["/O2", "/openmp", "/std:c++17", "-DENABLE_BF16"]
else:
    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
NVCC_FLAGS_COMMON = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    f"--threads={os.cpu_count()}",
    # "-Xptxas=-v",
    "-diag-suppress=174", # suppress the specific warning
    "-diag-suppress=177",
    "-diag-suppress=221",
]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS_COMMON += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

# Try to find CUDA_HOME if it's not set
if CUDA_HOME is None:
    # Check common CUDA installation paths
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.9",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12.0",
        os.getenv("CUDA_HOME"),
        os.getenv("CUDA_ROOT"),
    ]
    
    for cuda_path in cuda_paths:
        if cuda_path and os.path.exists(cuda_path):
            nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
            if os.path.isfile(nvcc_path):
                os.environ["CUDA_HOME"] = cuda_path
                print(f"Found CUDA at: {cuda_path}")
                break
            else:
                print(f"Found CUDA directory at {cuda_path} but nvcc compiler not found")
    else:
        raise RuntimeError(
            "Cannot find CUDA_HOME with nvcc compiler. CUDA toolkit must be installed to build the package. "
            "Tried paths: /usr/local/cuda, /usr/local/cuda-12.9, /usr/local/cuda-12.8, "
            "/usr/local/cuda-12.4, /usr/local/cuda-12.3, /usr/local/cuda-12.0")
    
    # Update CUDA_HOME variable
    CUDA_HOME = os.environ["CUDA_HOME"]

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_path = os.path.join(cuda_dir, "bin", "nvcc")
    if not os.path.isfile(nvcc_path):
        raise RuntimeError(f"nvcc not found at {nvcc_path}. CUDA compiler is required to build the package.")
    
    nvcc_output = subprocess.check_output([nvcc_path, "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

compute_capabilities = set()
if os.getenv("TORCH_CUDA_ARCH_LIST"):
    # TORCH_CUDA_ARCH_LIST is separated by space or semicolon
    for x in os.getenv("TORCH_CUDA_ARCH_LIST").replace(";", " ").split():
        compute_capabilities.add(x)
else:
    # Iterate over all GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
            continue
        compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    raise RuntimeError("No GPUs found. Please specify TORCH_CUDA_ARCH_LIST or build on a machine with GPUs.")
else:
    print(f"Detected compute capabilities: {compute_capabilities}")

def get_build_tag():
    """Generate build tag with PyTorch and CUDA versions for wheel naming (PEP 427 compliant)."""
    try:
        # Try to get versions from environment variables first (Docker build args)
        torch_version = os.getenv('TORCH_VERSION')
        cuda_version_str = os.getenv('CUDA_VERSION')
        
        # Fallback to detecting from installed PyTorch/CUDA if not in env vars
        if not torch_version:
            torch_version = torch.__version__.split('+')[0]  # Remove any +cu suffix
        if not cuda_version_str:
            cuda_version_str = str(nvcc_cuda_version)  # e.g., "12.9.1"
            
        print(f"Using PyTorch version: {torch_version}")
        print(f"Using CUDA version: {cuda_version_str}")
        
        if torch_version and cuda_version_str:
            # Convert to compact format: 2.8.0 -> torch280, 12.9.1 -> cu129
            torch_ver = torch_version.replace('.', '')  # "280"
            cuda_ver = '.'.join(cuda_version_str.split('.')[:2]).replace('.', '')  # "129"
            
            build_tag = f"{torch_ver}.{cuda_ver}"
            
            # Add optional wheel version suffix from environment
            wheel_suffix = os.environ.get("SAGEATTENTION_WHEEL_VERSION_SUFFIX", "")
            if wheel_suffix:
                # Remove leading + if present for build tag format
                wheel_suffix = wheel_suffix.lstrip('+')
                build_tag += f".{wheel_suffix}"
                print(f"Added wheel suffix: {wheel_suffix}")
            
            print(f"Generated build tag: {build_tag}")
            return build_tag
        else:
            print("Warning: Missing PyTorch or CUDA version information")
    except Exception as e:
        print(f"Warning: Could not generate build tag: {e}")
        import traceback
        traceback.print_exc()
    
    return ""

def has_capability(target):
    return any(cc.startswith(target) for cc in compute_capabilities)

# Validate the NVCC CUDA version.
if nvcc_cuda_version < Version("12.0"):
    raise RuntimeError("CUDA 12.0 or higher is required to build the package.")
if nvcc_cuda_version < Version("12.4") and has_capability("8.9"):
    raise RuntimeError(
        "CUDA 12.4 or higher is required for compute capability 8.9.")
if nvcc_cuda_version < Version("12.3") and has_capability("9.0"):
    raise RuntimeError(
        "CUDA 12.3 or higher is required for compute capability 9.0.")
if nvcc_cuda_version < Version("12.8") and has_capability("12.0"):
    raise RuntimeError(
        "CUDA 12.8 or higher is required for compute capability 12.0.")

# Add target compute capabilities to NVCC flags.
def get_nvcc_flags(allowed_capabilities):
    NVCC_FLAGS = []
    for capability in compute_capabilities:
        if capability not in allowed_capabilities:
            continue

        # capability: "8.0+PTX" -> num: "80"
        num = capability.split("+")[0].replace(".", "")
        if num in {"90", "120"}:
            # need to use sm90a instead of sm90 to use wgmma ptx instruction.
            # need to use sm120a to use mxfp8/mxfp4/nvfp4 instructions.
            num += "a"

        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

    NVCC_FLAGS += NVCC_FLAGS_COMMON
    return NVCC_FLAGS

ext_modules = []

if has_capability(("8.0",)):
    qattn_extension = CUDAExtension(
        name="sageattention._qattn_sm80",
        sources=[
            "csrc/qattn/pybind_sm80.cpp",
            "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": get_nvcc_flags(["8.0"]),
        },
    )
    ext_modules.append(qattn_extension)

if has_capability(("8.9", "12.0")):
    qattn_extension = CUDAExtension(
        name="sageattention._qattn_sm89",
        sources=[
            "csrc/qattn/pybind_sm89.cpp",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu"
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": get_nvcc_flags(["8.9", "12.0"]),
        },
    )
    ext_modules.append(qattn_extension)

if has_capability(("9.0",)):
    qattn_extension = CUDAExtension(
        name="sageattention._qattn_sm90",
        sources=[
            "csrc/qattn/pybind_sm90.cpp",
            "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
        ],
        libraries=["cuda"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": get_nvcc_flags(["9.0"]),
        },
    )
    ext_modules.append(qattn_extension)

# Fused kernels.
fused_extension = CUDAExtension(
    name="sageattention._fused",
    sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": get_nvcc_flags(["8.0", "8.9", "9.0", "12.0"]),
    },
)
ext_modules.append(fused_extension)


parallel = None
if 'EXT_PARALLEL' in os.environ:
    try:
        parallel = int(os.getenv('EXT_PARALLEL'))
    finally:
        pass


# Prevent file conflicts when multiple extensions are compiled simultaneously
class BuildExtensionSeparateDir(BuildExtension):
    build_extension_patch_lock = threading.Lock()
    thread_ext_name_map = {}

    def finalize_options(self):
        if parallel is not None:
            self.parallel = parallel
        super().finalize_options()

    def build_extension(self, ext):
        with self.build_extension_patch_lock:
            if not getattr(self.compiler, "_compile_separate_output_dir", False):
                compile_orig = self.compiler.compile

                def compile_new(*args, **kwargs):
                    return compile_orig(*args, **{
                        **kwargs,
                        "output_dir": os.path.join(
                            kwargs["output_dir"],
                            self.thread_ext_name_map[threading.current_thread().ident]),
                    })
                self.compiler.compile = compile_new
                self.compiler._compile_separate_output_dir = True
        self.thread_ext_name_map[threading.current_thread().ident] = ext.name
        objects = super().build_extension(ext)
        return objects


class bdist_wheel_with_build_tag(bdist_wheel):
    """Custom bdist_wheel command that sets build tag for PEP 427 compliance."""
    
    def initialize_options(self):
        super().initialize_options()
        # Set build tag from our function
        build_tag = get_build_tag()
        if build_tag:
            self.build_number = build_tag


setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtensionSeparateDir,
        "bdist_wheel": bdist_wheel_with_build_tag
    } if ext_modules else {"bdist_wheel": bdist_wheel_with_build_tag},
)
