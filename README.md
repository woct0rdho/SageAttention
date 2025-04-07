# [SageAttention](https://github.com/thu-ml/SageAttention) fork for build system integration

This repo makes it easy to build SageAttention for multiple Python, PyTorch, and CUDA versions, then distribute the wheels to other people. See [releases](https://github.com/woct0rdho/SageAttention/releases) for the wheels, and the [workflow](https://github.com/woct0rdho/SageAttention/blob/main/.github/workflows/build-sageattn.yml) to build them on Windows.

If you only need to build and run on your own machine, you can clone this repo, install the dependencies in [`pyproject.toml`](https://github.com/woct0rdho/SageAttention/blob/main/pyproject.toml) (include the correct torch version such as `torch 2.7.1+cu128`), then run `python setup.py install` (this avoids the environment checks of pip).
