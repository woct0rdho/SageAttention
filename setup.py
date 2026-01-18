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
import sys
import subprocess
import threading
import warnings
from packaging.version import parse, Version

from setuptools import setup, find_packages

# Skip CUDA build in CI or when explicitly requested
SKIP_CUDA_BUILD = (
    os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() in {"1", "TRUE", "YES"}
    or ("sdist" in sys.argv)
)

ext_modules = []
cmdclass = {}

if not SKIP_CUDA_BUILD:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    # Compiler flags.
    if os.name == "nt":
        # TODO: Detect MSVC rather than OS
        CXX_FLAGS = ["/O2", "/openmp", "/std:c++17", "/permissive-", "-DENABLE_BF16"]
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
        "-diag-suppress=174",
        "-diag-suppress=177",
        "-diag-suppress=221",
    ]
    if os.name == "nt":
        # https://github.com/pytorch/pytorch/issues/148317
        NVCC_FLAGS_COMMON += [
            "-D_WIN32=1",
            "-DUSE_CUDA=1",
        ]

    # Append flags from env if provided
    cxx_append = os.getenv("CXX_APPEND_FLAGS", "").strip()
    if cxx_append:
        CXX_FLAGS += cxx_append.split()
    nvcc_append = os.getenv("NVCC_APPEND_FLAGS", "").strip()
    if nvcc_append:
        NVCC_FLAGS_COMMON += nvcc_append.split()

    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS_COMMON += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

    if CUDA_HOME is None:
        raise RuntimeError(
            "Cannot find CUDA_HOME. CUDA must be available to build the package.")

    def get_nvcc_cuda_version(cuda_dir: str) -> Version:
        """Get the CUDA version from nvcc.

        Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
        """
        nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                              universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        return nvcc_cuda_version

    # Determine target compute capabilities
    compute_capabilities = set()
    if os.getenv("TORCH_CUDA_ARCH_LIST"):
        # Prefer TORCH_CUDA_ARCH_LIST if explicitly specified (works without GPUs)
        # TORCH_CUDA_ARCH_LIST is separated by space or semicolon
        for x in os.getenv("TORCH_CUDA_ARCH_LIST").replace(";", " ").split():
            compute_capabilities.add(x)
    else:
         # If not provided, try to detect from local GPUs
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for i in range(device_count):
            major, minor = torch.cuda.get_device_capability(i)
            if major < 7:
                warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
                continue
            compute_capabilities.add(f"{major}.{minor}")

    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)

    if not compute_capabilities:
        raise RuntimeError(
            "No target compute capabilities. Set TORCH_CUDA_ARCH_LIST or build on a machine with GPUs.")
    else:
        print(f"Target compute capabilities: {compute_capabilities}")

    def has_capability(target):
        return any(cc.startswith(target) for cc in compute_capabilities)

    # Validate the NVCC CUDA version.
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
            if num in {"90", "100", "120", "121"}:
                # need to use sm90a instead of sm90 to use wgmma ptx instruction.
                # need to use sm120a to use mxfp8/mxfp4/nvfp4 instructions.
                num += "a"

            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
            if capability.endswith("+PTX"):
                NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]
        NVCC_FLAGS += NVCC_FLAGS_COMMON
        return NVCC_FLAGS

    if has_capability(("8.0", "8.6", "8.7")):
        ext_modules.append(
            CUDAExtension(
                name="sageattention._qattn_sm80",
                sources=[
                    "csrc/qattn/pybind_sm80.cpp",
                    "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
                ],
                extra_compile_args={
                    "cxx": CXX_FLAGS,
                    # Build binary for sm80 if sm86/87 is detected. No need to build binary for sm86/87
                    "nvcc": get_nvcc_flags(["8.0"]),
                },
            )
        )

    if has_capability(("8.9", "10.0", "12.0", "12.1")):
        ext_modules.append(
            CUDAExtension(
                name="sageattention._qattn_sm89",
                sources=[
                    "csrc/qattn/pybind_sm89.cpp",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
                ],
                extra_compile_args={
                    "cxx": CXX_FLAGS,
                    "nvcc": get_nvcc_flags(["8.9", "10.0", "12.0", "12.1"]),
                },
            )
        )

    if has_capability("9.0"):
        ext_modules.append(
            CUDAExtension(
                name="sageattention._qattn_sm90",
                sources=[
                    "csrc/qattn/pybind_sm90.cpp",
                    "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
                ],
                libraries=["cuda"],
                library_dirs=[os.path.join(CUDA_HOME, "lib64", "stubs")],
                extra_compile_args={
                    "cxx": CXX_FLAGS,
                    "nvcc": get_nvcc_flags(["9.0"]),
                },
            )
        )

    ext_modules.append(
        CUDAExtension(
            name="sageattention._fused",
            sources=[
                "csrc/fused/pybind.cpp",
                "csrc/fused/fused.cu",
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": get_nvcc_flags(["8.0", "8.9", "9.0", "10.0", "12.0", "12.1"]),
            },
        )
    )

    # Resolve parallelism from env
    parallel = None
    if 'EXT_PARALLEL' in os.environ:
        parallel = int(os.getenv('EXT_PARALLEL'))
    if parallel is None and 'MAX_JOBS' in os.environ:
        parallel = int(os.getenv('MAX_JOBS'))
    # Defaults if not provided
    if parallel is None:
        parallel = os.cpu_count()
    # Ensure MAX_JOBS for underlying tooling if not explicitly set
    os.environ.setdefault('MAX_JOBS', str(parallel))

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

    cmdclass = {"build_ext": BuildExtensionSeparateDir} if ext_modules else {}

setup(
    name='sageattention',
    version='2.2.0',
    author='SageAttention team',
    license='Apache 2.0 License',
    description='Accurate and efficient plug-and-play low-bit attention.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/SageAttention',
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
