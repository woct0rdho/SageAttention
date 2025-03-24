# Add torch version in pyproject.toml according to the environment variables

import os

with open("./pyproject.toml", "r") as f:
    text = f.read()

TORCH_MINOR_VERSION = os.getenv("TORCH_MINOR_VERSION", "6")
TORCH_PATCH_VERSION = os.getenv("TORCH_PATCH_VERSION", "0")
TORCH_PATCH_VERSION_NEXT = str(int(TORCH_PATCH_VERSION) + 1)
TORCH_VERSION = f"2.{TORCH_MINOR_VERSION}.{TORCH_PATCH_VERSION}"
TORCH_VERSION_NEXT = f"2.{TORCH_MINOR_VERSION}.{TORCH_PATCH_VERSION_NEXT}"
text = text.replace('"torch"', f'"torch>={TORCH_VERSION},<{TORCH_VERSION_NEXT}"')

with open("./pyproject.toml", "w") as f:
    f.write(text)
