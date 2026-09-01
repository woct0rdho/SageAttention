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


def get_git_commit_timestamp():
    try:
        timestamp = subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "log", "-1", "--format=%ct"]
        ).strip().decode("utf-8")
    except Exception:
        return None
    return timestamp if timestamp.isdigit() else None


# Make wheel ZIP metadata deterministic
# If SOURCE_DATE_EPOCH is unspecified, then query with git, then fallback to Unix epoch
os.environ.setdefault("SOURCE_DATE_EPOCH", get_git_commit_timestamp() or "315532800")

# Skip CUDA build in CI or when explicitly requested
SKIP_CUDA_BUILD = (
    os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() in {"1", "TRUE", "YES"}
    or ("sdist" in sys.argv)
)

ext_modules = []
cmdclass = {}


def append_env_flags(flags, env_name):
    extra = os.getenv(env_name, "").strip()
    if extra:
        flags += extra.split()


def unique_paths(paths):
    result = []
    seen = set()
    for path_value in paths:
        if path_value and path_value not in seen:
            result.append(path_value)
            seen.add(path_value)
    return result


def rocm_sdk_path(which):
    try:
        return subprocess.check_output(
            ["rocm-sdk", "path", f"--{which}"], text=True
        ).strip()
    except Exception:
        return None


def configure_rocm(default_rocm_home):
    sdk_root = rocm_sdk_path("root")
    sdk_bin = rocm_sdk_path("bin")
    rocm_home = sdk_root or default_rocm_home or os.getenv("ROCM_HOME")
    if not rocm_home:
        raise RuntimeError(
            "Cannot find ROCm. Activate a ROCm-enabled PyTorch environment."
        )

    os.environ["ROCM_HOME"] = rocm_home
    if os.name == "nt":
        os.environ.setdefault("CC", "clang-cl")
        os.environ.setdefault("CXX", "clang-cl")
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")

    path_parts = [
        os.path.join(rocm_home, "lib", "llvm", "bin"),
        os.path.join(rocm_home, "bin"),
        sdk_bin,
    ]
    os.environ["PATH"] = os.pathsep.join(
        unique_paths(path_parts) + [os.environ.get("PATH", "")]
    )
    return rocm_home


def rocm_arches(torch):
    arch_env = os.getenv("GPU_ARCHS") or os.getenv("PYTORCH_ROCM_ARCH")
    if arch_env:
        archs = [
            arch.split(":", 1)[0]
            for arch in arch_env.replace(";", " ").replace(",", " ").split()
            if arch.strip()
        ]
    else:
        archs = []
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(device_idx)
                arch = getattr(props, "gcnArchName", "")
                if arch:
                    archs.append(arch.split(":", 1)[0])
    return sorted(set(archs))


if not SKIP_CUDA_BUILD:
    import torch
    import torch.utils.cpp_extension as cpp_extension
    from torch.utils.cpp_extension import (
        BuildExtension,
        CUDAExtension,
        CUDA_HOME,
        ROCM_HOME,
    )

    def add_windows_reproducible_path_flags(
        cxx_flags, device_flags, path_mappings, *, clang_driver=False
    ):
        if os.name != "nt":
            return

        if clang_driver:
            cxx_flags.append("/Brepro")
            device_flags.append("-frandom-seed=0")
            for source, target in path_mappings:
                source = os.path.normpath(os.path.realpath(source))
                prefix_map = f"-ffile-prefix-map={source}={target}"
                cxx_flags.append(f"/clang:{prefix_map}")
                device_flags.append(prefix_map)
            return

        cxx_flags.append("/experimental:deterministic")
        device_flags.extend(["-Xcompiler", "/experimental:deterministic"])

        for source, target in path_mappings:
            source = os.path.normpath(os.path.realpath(source))
            cxx_flags.append(f"/pathmap:{source}={target}")
            device_flags.extend(["-Xcompiler", f"/pathmap:{source}={target}"])

    if torch.version.hip is not None:
        rocm_home = configure_rocm(ROCM_HOME)
        cpp_extension.ROCM_HOME = rocm_home

        amd_arches = rocm_arches(torch) or ["gfx1200", "gfx1201"]
        os.environ["PYTORCH_ROCM_ARCH"] = ";".join(amd_arches)
        print(f"Target AMD GPU architectures: {amd_arches}")

        limited_api_flags = [
            "-DPy_LIMITED_API=0x030A0000",
            "-DTORCH_STABLE_ONLY",
        ]
        if os.name == "nt":
            cxx_flags = ["/O2", "/permissive-", "-DENABLE_BF16"]
            link_flags = ["/Brepro"]
        else:
            abi = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
            cxx_flags = [
                "-O3",
                "-DENABLE_BF16",
                f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
            ]
            link_flags = []
        cxx_flags += limited_api_flags

        hip_flags = [
            "-O3",
            "-ffast-math",
            "-fgpu-flush-denormals-to-zero",
            "-fno-offload-uniform-block",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-mllvm",
            "--lsr-drop-solution=1",
            "-mllvm",
            "-enable-post-misched=1",
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-amdgpu-function-calls=false",
        ] + limited_api_flags
        if os.name == "nt":
            # Avoid injecting Clang's HIP wrapper before MSVC/PyTorch headers.
            # HIP translation units include the wrapper headers explicitly in
            # an order compatible with VS 2026's <cmath>.
            hip_flags.append("-nohipwrapperinc")
        else:
            hip_flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={abi}")
        for arch in amd_arches:
            hip_flags.append(f"--offload-arch={arch}")
        hip_flags.append(f"--rocm-path={rocm_home}")

        rocm_device_lib_path = os.path.join(
            rocm_home, "lib", "llvm", "amdgcn", "bitcode"
        )
        if os.path.isdir(rocm_device_lib_path):
            hip_flags.append(
                f"--rocm-device-lib-path={rocm_device_lib_path}"
            )

        append_env_flags(cxx_flags, "CXX_APPEND_FLAGS")
        append_env_flags(hip_flags, "NVCC_APPEND_FLAGS")
        append_env_flags(hip_flags, "HIPCC_APPEND_FLAGS")

        add_windows_reproducible_path_flags(
            cxx_flags,
            hip_flags,
            [
                (os.path.dirname(os.path.abspath(__file__)), r"C:\reproducible\path\SageAttention"),
                (os.path.dirname(os.path.abspath(torch.__file__)), r"C:\reproducible\path\torch"),
            ],
            clang_driver=True,
        )

        include_dirs = unique_paths([os.path.join(rocm_home, "include")])

        if any(arch.startswith("gfx12") for arch in amd_arches):
            ext_modules.append(
                CUDAExtension(
                    name="sageattention._qattn_gfx12_native",
                    sources=[
                        "csrc/qattn/pybind_gfx12_native.cpp",
                        "csrc/qattn/qk_int_sv_gfx12_native_aux.cu",
                        "csrc/qattn/qk_int_sv_gfx12_native_prepare.cu",
                        "csrc/qattn/qk_int_sv_gfx12_native_attn_f16.cu",
                        "csrc/qattn/qk_int_sv_gfx12_native_attn_fp8.cu",
                        "csrc/qattn/qk_int_sv_gfx12_native_rawq_fp8.cu",
                    ],
                    include_dirs=include_dirs,
                    extra_compile_args={
                        "cxx": cxx_flags,
                        "nvcc": hip_flags,
                    },
                    extra_link_args=link_flags,
                    py_limited_api=True,
                )
            )
        else:
            warnings.warn(
                "ROCm build detected, but no gfx12 architecture was selected; "
                "skipping the gfx12 native attention extension."
            )

        ext_modules.append(
            CUDAExtension(
                name="sageattention._fused",
                sources=[
                    "csrc/fused/pybind.cpp",
                    "csrc/fused/fused.cu",
                ],
                include_dirs=include_dirs,
                extra_compile_args={
                    "cxx": cxx_flags,
                    "nvcc": hip_flags,
                },
                extra_link_args=link_flags,
                py_limited_api=True,
            )
        )
    else:
        # Compiler flags.
        if os.name == "nt":
            # TODO: Detect MSVC rather than OS
            CXX_FLAGS = ["/O2", "/permissive-", "-DENABLE_BF16"]
            LINK_FLAGS = ["/Brepro"]
        else:
            CXX_FLAGS = ["-O3", "-DENABLE_BF16"]
            LINK_FLAGS = []
        CXX_FLAGS += ["-DPy_LIMITED_API=0x030A0000", "-DTORCH_STABLE_ONLY"]

        NVCC_FLAGS_COMMON = [
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--use_fast_math",
            f"--threads={os.cpu_count()}",
            # "-Xptxas=-v",
            "-diag-suppress=174",
            "-diag-suppress=177",
            "-diag-suppress=221",
            "-DPy_LIMITED_API=0x030A0000",
            "-DTORCH_STABLE_ONLY",
        ]
        if os.name == "nt":
            # https://github.com/pytorch/pytorch/issues/148317
            NVCC_FLAGS_COMMON += [
                "-Xcompiler=/Zc:preprocessor",
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

        if os.name != "nt":
            ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
            CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
            NVCC_FLAGS_COMMON += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

        if CUDA_HOME is None:
            raise RuntimeError(
                "Cannot find CUDA_HOME. CUDA must be available to build the package.")

        add_windows_reproducible_path_flags(
            CXX_FLAGS,
            NVCC_FLAGS_COMMON,
            [
                (os.path.dirname(os.path.abspath(__file__)), r"C:\reproducible\path\SageAttention"),
                (os.path.dirname(os.path.abspath(torch.__file__)), r"C:\reproducible\path\torch"),
            ],
        )

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

        def capability_sort_key(capability):
            base = capability.split("+")[0]
            major, minor = base.split(".")
            return (int(major), int(minor), capability)

        # Sort compute_capabilities for reproducible build
        compute_capabilities = sorted(compute_capabilities, key=capability_sort_key)

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
                    extra_link_args=LINK_FLAGS,
                    py_limited_api=True,
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
                    extra_link_args=LINK_FLAGS,
                    py_limited_api=True,
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
                    extra_compile_args={
                        "cxx": CXX_FLAGS,
                        "nvcc": get_nvcc_flags(["9.0"]),
                    },
                    extra_link_args=LINK_FLAGS,
                    py_limited_api=True,
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
                extra_link_args=LINK_FLAGS,
                py_limited_api=True,
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
    version='2.2.0' + os.environ.get("SAGEATTENTION_WHEEL_VERSION_SUFFIX", ""),
    author='SageAttention team',
    license='Apache 2.0 License',
    description='Accurate and efficient plug-and-play low-bit attention.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/SageAttention',
    packages=find_packages(),
    python_requires='>=3.10',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
