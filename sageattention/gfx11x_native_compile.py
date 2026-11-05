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

import torch

# Importing this module loads the _qattn_gfx110x extension, whose
# STABLE_TORCH_LIBRARY static initializers register the
# sageattention_qattn_gfx110x ops.
from . import _qattn_gfx110x  # noqa: F401

# Re-export the ops namespace so callers use torch.ops.sageattention_qattn_gfx110x.
_qattn_gfx110x = torch.ops.sageattention_qattn_gfx110x  # noqa: F811