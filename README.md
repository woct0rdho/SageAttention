# [SageAttention](https://github.com/thu-ml/SageAttention) fork for Windows wheels and easy installation

This repo makes it easy to build SageAttention for multiple Python, PyTorch, and CUDA versions, then distribute the wheels to other people.

## Know before installing

First, note that if you just `pip install sageattention`, that's actually [SageAttention 1](https://pypi.org/project/sageattention/#history), which uses Triton and no CUDA and is easy to install.

Here is SageAttention 2, which has both Triton and CUDA kernels, and can be faster than SageAttention 1 in many cases.

The latest wheels support GTX 16xx, RTX 20xx/30xx/40xx/50xx, A100, H100 (sm75/80/86/89/90/120). 

## Installation

1. Know how to use pip to install packages in the correct Python environment, see https://github.com/woct0rdho/triton-windows
2. Install triton-windows
3. Install a wheel in the release page: https://github.com/woct0rdho/SageAttention/releases
    * Unlike triton-windows, you need to manually choose a wheel in the GitHub release page for SageAttention
    * Choose the wheel for your PyTorch version. For example, 'torch2.7.0' in the filename
        * The torch minor version (2.6/2.7 ...) must be correct, but the patch version (2.7.0/2.7.1 ...) can be different from yours
    * No need to worry about tbe CUDA minor version (12.8/12.9 ...). It can be different from yours, because SageAttention does not yet use any breaking API
        * But there is a difference between CUDA 12 and 13
    * No need to worry about tbe Python minor version (3.10/3.11 ...). The recent wheels use Python Stable ABI (also known as ABI3) and have `cp39-abi3` in the filenames, so they support Python >= 3.9

If you see any error, please open an issue at https://github.com/woct0rdho/SageAttention/issues

Recently we've simplified the installation by a lot. There is no need to install Visual Studio or CUDA toolkit to use Triton and SageAttention (unless you want to step into the world of building from source)

## Use notes

Before using SageAttention in larger projects like ComfyUI, please run [test_sageattn.py](https://github.com/woct0rdho/SageAttention/blob/main/tests/test_sageattn.py) to test if SageAttention itself works.

To use SageAttention in ComfyUI, you just need to add `--use-sage-attention` when starting ComfyUI.

Some recent models, such as Wan and Qwen-Image, may produce black or noise output when SageAttention is used, because some intermediate values overflow SageAttention's quantization. In this case, you may use the `PatchSageAttentionKJ` node in KJNodes, and choose `sageattn_qk_int8_pv_fp16_cuda`, which is the least likely to overflow.

Also, if you want to run Flux or Qwen-Image, try [Nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) if you haven't. It's faster and more accurate than GGUF Q4_0 + SageAttention.

If you want to run Wan, try [RadialAttention](https://github.com/woct0rdho/ComfyUI-RadialAttn) if you haven't. It's also faster than SageAttention.

## Build from source

(This is for developers)

If you need to build and run SageAttention on your own machine:
1. Install Visual Studio (MSVC and Windows SDK), and CUDA toolkit
2. Clone this repo. Checkout `abi3_stable` branch if you want ABI3 and libtorch stable ABI
3. Install the dependencies in [`pyproject.toml`](https://github.com/woct0rdho/SageAttention/blob/main/pyproject.toml), include the correct torch version such as `torch 2.7.1+cu128`
4. Run `python setup.py install --verbose` to install directly, or `python setup.py bdist_wheel --verbose` to build a wheel. This avoids the environment checks of pip

## Dev notes

* The wheels are built using the [workflow](https://github.com/woct0rdho/SageAttention/blob/main/.github/workflows/build-sageattn.yml)
    * It's tricky to specify both torch (with index URL at download.pytorch.org ) and pybind11 (not in that index URL) in an isolated build environment. The easiest way I could think of is to use [simpleindex](https://github.com/uranusjr/simpleindex)
* CUDA kernels for sm80/86/89/90 are bundled in the wheels, and also sm120 for CUDA >= 12.8
* For RTX 20xx, SageAttention 2 runs Triton kernels, which are the same as SageAttention 1. If you want to help improve the CUDA kernels for RTX 20xx, you may see https://github.com/Ph0rk0z/SageAttention2/tree/updates
* The wheels do not use CXX11 ABI
* We cannot publish the wheels to PyPI, because PyPI does not support multiple PyTorch/CUDA variants for the same version of SageAttention. Some people are working on this, see https://astral.sh/blog/introducing-pyx and https://wheelnext.dev/proposals/pepxxx_wheel_variant_support/
