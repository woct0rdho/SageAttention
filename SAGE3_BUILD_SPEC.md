# SageAttention3 Build Specification

This document defines the requirements and design for SageAttention3 (Blackwell-only) wheel builds.

## Executive Summary

SageAttention3 is a specialized build targeting Blackwell GPU architecture (B100/B200) with FP4 Microscaling attention. It requires stricter hardware and software requirements compared to SageAttention 2/2++, necessitating separate build workflows.

---

## Requirements Analysis

### Hard Requirements (from setup.py)

1. **CUDA Version**: 12.8+ (raises RuntimeError if < 12.8)
2. **Compute Capabilities**: Only sm_100a (10.0) and/or sm_120a (12.0)
3. **Python Version**: 3.9+ (supports up to 3.14 per upstream workflow)
4. **PyTorch Version**: 2.7.0+ (CUDA 12.8 support), 2.8.0+ recommended
5. **Package Name**: `sageattn3` (different from main `sageattention` package)
6. **Build Directory**: `sageattention3_blackwell/` subdirectory
7. **Submodules**: Requires `git submodule update --init --recursive` for CUTLASS

### Key Differences from SageAttention2

| Aspect | SageAttention 2/2++ | SageAttention3 |
|--------|---------------------|----------------|
| **Package** | `sageattention` | `sageattn3` |
| **Directory** | `.` (root) | `sageattention3_blackwell/` |
| **CUDA Min** | 12.0+ | 12.8+ |
| **GPU Arch** | sm70-sm121 (10 archs) | sm100a, sm120a only |
| **GPU Support** | V100→B200 (broad) | B100/B200 only (Blackwell) |
| **Python** | 3.9-3.12 | 3.9-3.14 |
| **PyTorch** | 2.6.0-2.8.0 | 2.7.0-2.8.0+ |
| **Technology** | INT8/FP16/FP8 | FP4 Microscaling |
| **Wheel Count** | 10 combinations | 6-8 combinations |

---

## Build Matrix Design

### Supported Combinations

#### PyTorch & CUDA Compatibility

| PyTorch Version | CUDA Version | Reason |
|----------------|--------------|---------|
| 2.7.0 | 12.8.1 | Minimum CUDA 12.8 requirement |
| 2.8.0 | 12.9.1 | Latest stable with CUDA 12.9 |

**Note**: PyTorch 2.6 not supported (CUDA 12.6 < 12.8 minimum)

#### Python Support

| Python Version | Status | Notes |
|---------------|--------|-------|
| 3.9 | ✅ Supported | Minimum version |
| 3.10 | ✅ Supported | Stable |
| 3.11 | ✅ Supported | Stable |
| 3.12 | ✅ Supported | Recommended |
| 3.13 | ✅ Supported | Latest stable |
| 3.14 | 🔄 Future | When available |

### Complete Build Matrix

**Total: 8 wheel combinations**

| # | PyTorch | CUDA | Python | Build Target |
|---|---------|------|--------|--------------|
| 1 | 2.7.0 | 12.8.1 | 3.10 | `sage3-pytorch27-cu128-python310` |
| 2 | 2.7.0 | 12.8.1 | 3.11 | `sage3-pytorch27-cu128-python311` |
| 3 | 2.7.0 | 12.8.1 | 3.12 | `sage3-pytorch27-cu128-python312` |
| 4 | 2.7.0 | 12.8.1 | 3.13 | `sage3-pytorch27-cu128-python313` |
| 5 | 2.8.0 | 12.9.1 | 3.10 | `sage3-pytorch28-cu129-python310` |
| 6 | 2.8.0 | 12.9.1 | 3.11 | `sage3-pytorch28-cu129-python311` |
| 7 | 2.8.0 | 12.9.1 | 3.12 | `sage3-pytorch28-cu129-python312` |
| 8 | 2.8.0 | 12.9.1 | 3.13 | `sage3-pytorch28-cu129-python313` |

### Build Profiles

Similar to SageAttention2 workflows, but with fewer combinations:

| Profile | Description | Wheel Count | Combinations |
|---------|-------------|-------------|--------------|
| `default` | Single latest build | 1 | PyTorch 2.8 + CUDA 12.9 + Python 3.12 |
| `stable` | Latest Python per PyTorch | 2 | PyTorch 2.7 + Py3.13, PyTorch 2.8 + Py3.13 |
| `pytorch27` | All Python for PyTorch 2.7 | 4 | Python 3.10-3.13 |
| `pytorch28` | All Python for PyTorch 2.8 | 4 | Python 3.10-3.13 |
| `python310` | PyTorch 2.7-2.8 for Py3.10 | 2 | Both PyTorch versions |
| `python311` | PyTorch 2.7-2.8 for Py3.11 | 2 | Both PyTorch versions |
| `python312` | PyTorch 2.7-2.8 for Py3.12 | 2 | Both PyTorch versions |
| `python313` | PyTorch 2.7-2.8 for Py3.13 | 2 | Both PyTorch versions |
| `sage3-all` | All combinations | 8 | All listed above |

---

## Wheel Naming Convention

### Format (PEP 427 Compliant)

```
sageattn3-{version}-{build_tag}-cp{python}-cp{python}-{platform}.whl
```

### Components

| Component | Format | Example | Description |
|-----------|--------|---------|-------------|
| `{version}` | Semantic version | `1.0.0` | SageAttention3 package version |
| `{build_tag}` | `{torch}.{cuda}` | `280.129` | PyTorch 2.8.0 + CUDA 12.9 |
| `{python}` | `cp{version}` | `cp312` | Python 3.12 |
| `{abi}` | `cp{version}` | `cp312` | CPython ABI tag |
| `{platform}` | Platform tag | `linux_x86_64` or `win_amd64` | Platform |

### Examples

```bash
# Linux wheels
sageattn3-1.0.0-270.128-cp312-cp312-linux_x86_64.whl  # PyTorch 2.7.0 + CUDA 12.8 + Python 3.12
sageattn3-1.0.0-280.129-cp313-cp313-linux_x86_64.whl  # PyTorch 2.8.0 + CUDA 12.9 + Python 3.13

# Windows wheels
sageattn3-1.0.0-270.128-cp312-cp312-win_amd64.whl     # PyTorch 2.7.0 + CUDA 12.8 + Python 3.12
sageattn3-1.0.0-280.129-cp313-cp313-win_amd64.whl     # PyTorch 2.8.0 + CUDA 12.9 + Python 3.13
```

### Build Tag Decoding

| Build Tag | PyTorch | CUDA | Full Versions |
|-----------|---------|------|---------------|
| `270.128` | 2.7 | 12.8 | PyTorch 2.7.0 + CUDA 12.8.1 |
| `280.129` | 2.8 | 12.9 | PyTorch 2.8.0 + CUDA 12.9.1 |

---

## Workflow Architecture

### Proposed Workflows

Two separate workflow files for clean separation:

1. **`build-wheels-sage3-linux.yml`**
   - Platform: Linux (Ubuntu + Docker Buildx)
   - Build method: Docker multi-stage builds
   - Working directory: `sageattention3_blackwell/`
   - Outputs: `linux_x86_64` wheels

2. **`build-wheels-sage3-windows.yml`**
   - Platform: Windows (Native MSVC)
   - Build method: Native Windows builds
   - Working directory: `sageattention3_blackwell/`
   - Outputs: `win_amd64` wheels

### Workflow Triggers

Both workflows should support:

1. **Pull Requests**: Build `default` profile (1 wheel for testing)
2. **Main Branch Push**: Build `stable` profile (2 wheels)
3. **Git Tags (`v*-sage3`)**: Build `sage3-all` profile (8 wheels) + create release
4. **Manual Dispatch**: Choose any build profile

**Note**: Use separate tag naming (`v1.0.0-sage3`) to avoid conflicts with SageAttention2 releases.

### Job Structure

Both workflows follow the same structure as SageAttention2:

```yaml
jobs:
  1. validate          # Determine build profile
  2. generate-matrix   # Create dynamic build matrix
  3. build-wheels      # Build wheels in parallel
  4. build-summary     # Aggregate results
  5. release           # Create GitHub Release (tags only)
```

---

## Build Environment Requirements

### CUDA Architecture Flags

```bash
TORCH_CUDA_ARCH_LIST="10.0;12.0"  # Both Blackwell architectures
```

**CUDA Compilation Flags** (from setup.py):
- Architecture: `-gencode arch=compute_100a,code=sm_100a` and/or `arch=compute_120a,code=sm_120a`
- Compiler flags include FP4 quantization optimizations
- Requires CUTLASS library (git submodule)

### Build Steps (Linux)

1. Checkout repository
2. Initialize git submodules (`git submodule update --init --recursive`)
3. Change to `sageattention3_blackwell/` directory
4. Setup Python environment
5. Install PyTorch for target CUDA version
6. Run `update_pyproject.py` to set PyTorch dependency
7. Build wheel: `python setup.py bdist_wheel`
8. Verify wheel naming and installation
9. Test import: `import sageattn3`

### Build Steps (Windows)

Same as Linux, but with:
- MSVC compiler setup
- N-Storm/cuda-toolkit action for CUDA installation
- PowerShell-based build scripts

---

## Release Strategy

### Tag Naming Convention

Use separate tag prefix to distinguish from SageAttention2:

```bash
# SageAttention2 tags
v2.2.0, v2.2.1, etc.

# SageAttention3 tags
v1.0.0-sage3, v1.0.1-sage3, etc.
```

### Release Workflow

When tag `v*-sage3` is pushed:

1. **Linux workflow** creates GitHub Release
2. **Windows workflow** appends wheels to existing release
3. Final release contains 16 wheels (8 Linux + 8 Windows)

### Release Assets

```
Release v1.0.0-sage3
├── sageattn3-1.0.0-270.128-cp310-cp310-linux_x86_64.whl
├── sageattn3-1.0.0-270.128-cp311-cp311-linux_x86_64.whl
├── sageattn3-1.0.0-270.128-cp312-cp312-linux_x86_64.whl
├── sageattn3-1.0.0-270.128-cp313-cp313-linux_x86_64.whl
├── sageattn3-1.0.0-280.129-cp310-cp310-linux_x86_64.whl
├── sageattn3-1.0.0-280.129-cp311-cp311-linux_x86_64.whl
├── sageattn3-1.0.0-280.129-cp312-cp312-linux_x86_64.whl
├── sageattn3-1.0.0-280.129-cp313-cp313-linux_x86_64.whl
├── sageattn3-1.0.0-270.128-cp310-cp310-win_amd64.whl
├── sageattn3-1.0.0-270.128-cp311-cp311-win_amd64.whl
├── sageattn3-1.0.0-270.128-cp312-cp312-win_amd64.whl
├── sageattn3-1.0.0-270.128-cp313-cp313-win_amd64.whl
├── sageattn3-1.0.0-280.129-cp310-cp310-win_amd64.whl
├── sageattn3-1.0.0-280.129-cp311-cp311-win_amd64.whl
├── sageattn3-1.0.0-280.129-cp312-cp312-win_amd64.whl
└── sageattn3-1.0.0-280.129-cp313-cp313-win_amd64.whl
```

**Total**: 16 wheels per release

---

## Testing Strategy

### Validation Checks

1. **Wheel naming validation**: Verify PEP 427 compliance
2. **Version extraction**: Parse and validate version components from filename
3. **Import test**: `python -c "import sageattn3; print('Success')"`
4. **CUDA version check**: Ensure CUDA 12.8+ requirement is enforced

### Test Matrix

For PR testing (default profile):
- Build 1 wheel: PyTorch 2.8.0 + CUDA 12.9 + Python 3.12
- Validate naming: `sageattn3-1.0.0-280.129-cp312-cp312-{platform}.whl`
- Test import on build environment

---

## Docker Support (Linux)

### Dockerfile Requirements

Create `dockerfile.builder.sage3` similar to `dockerfile.builder.linux`:

```dockerfile
# Key differences from SageAttention2 Dockerfile:
# - WORKDIR /src/sageattention3_blackwell
# - Must include git submodule initialization
# - CUDA_VERSION >= 12.8
# - TORCH_CUDA_ARCH_LIST="10.0;12.0"
# - Wheel outputs: sageattn3-*.whl
```

### Docker Bake Configuration

Option 1: **Extend existing `docker-bake.hcl`** with Sage3 targets:

```hcl
# New targets in docker-bake.hcl
target "linux-sage3-pytorch27-cu128-python312" { ... }
target "linux-sage3-pytorch28-cu129-python312" { ... }

# New groups
group "sage3-all" { targets = [...] }
group "sage3-stable" { targets = [...] }
```

Option 2: **Separate `docker-bake-sage3.hcl`** for cleaner separation.

**Recommendation**: Extend existing file with clear naming (`sage3-` prefix).

---

## Implementation Checklist

### Phase 1: Docker Configuration (Linux)
- [ ] Create `dockerfile.builder.sage3` for SageAttention3 builds
- [ ] Add Sage3 targets to `docker-bake.hcl` (or create `docker-bake-sage3.hcl`)
- [ ] Create Sage3 build groups (sage3-all, sage3-stable, etc.)
- [ ] Test local Docker builds

### Phase 2: Linux Workflow
- [ ] Create `.github/workflows/build-wheels-sage3-linux.yml`
- [ ] Implement validate job with profile selection
- [ ] Implement generate-matrix job with 8 build profiles
- [ ] Implement build-wheels job using Docker Bake
- [ ] Implement wheel naming validation
- [ ] Implement build-summary job
- [ ] Implement release job (tag `v*-sage3` only)
- [ ] Test PR trigger (default profile)
- [ ] Test main branch trigger (stable profile)

### Phase 3: Windows Workflow
- [ ] Create `.github/workflows/build-wheels-sage3-windows.yml`
- [ ] Mirror Linux workflow structure
- [ ] Use N-Storm/cuda-toolkit@v0.2.28 for CUDA installation
- [ ] Implement native Windows build steps
- [ ] Add git submodule initialization
- [ ] Set working directory to `sageattention3_blackwell/`
- [ ] Implement release append logic
- [ ] Test Windows builds

### Phase 4: Documentation
- [ ] Update `CI_WORKFLOW.md` with Sage3 workflows
- [ ] Update `BUILD_MATRIX.md` with Sage3 combinations
- [ ] Update `CLAUDE.md` with Sage3 build commands
- [ ] Create workflow comparison table (Sage2 vs Sage3)
- [ ] Document tag naming convention (`v*-sage3`)

### Phase 5: Validation & Testing
- [ ] Test full release workflow (16 wheels)
- [ ] Verify wheel naming consistency across platforms
- [ ] Validate CUDA 12.8+ requirement enforcement
- [ ] Test manual workflow dispatch with all profiles
- [ ] Verify GitHub Release creation and append logic

---

## Open Questions

1. **Versioning**: Should SageAttention3 use independent versioning (1.0.0) or match SageAttention2 (2.2.0)?
   - **Recommendation**: Independent (1.0.0) since it's a different package

2. **Python 3.14**: Include in matrix when available?
   - **Recommendation**: Add when PyTorch supports it

3. **Docker Bake**: Extend existing or create separate file?
   - **Recommendation**: Extend existing with `sage3-` prefix for clarity

4. **Release Tags**: Use `v*-sage3` or separate release strategy?
   - **Recommendation**: Use `v*-sage3` tags to avoid conflicts

5. **Workflow Triggers**: Should Sage3 workflows run on same triggers as Sage2?
   - **Recommendation**: Yes, but only on Sage3-specific tags for releases

---

## Success Criteria

- [ ] 16 wheels (8 Linux + 8 Windows) built successfully per release
- [ ] All wheels follow PEP 427 naming convention
- [ ] Wheel naming consistent across Linux and Windows
- [ ] CUDA 12.8+ requirement enforced at build time
- [ ] Import test passes: `import sageattn3`
- [ ] GitHub Release created with all wheels attached
- [ ] Documentation complete and accurate
- [ ] Workflows reuse existing patterns from SageAttention2

---

## Timeline Estimate

| Phase | Estimated Time | Dependencies |
|-------|---------------|--------------|
| Phase 1: Docker Config | 2-3 hours | None |
| Phase 2: Linux Workflow | 3-4 hours | Phase 1 |
| Phase 3: Windows Workflow | 2-3 hours | Phase 2 |
| Phase 4: Documentation | 1-2 hours | Phases 2-3 |
| Phase 5: Testing | 2-3 hours | All phases |
| **Total** | **10-15 hours** | |

---

## References

- [SageAttention3 setup.py](sageattention3_blackwell/setup.py) - Build configuration
- [build-sageattn3.yml](.github/workflows/build-sageattn3.yml) - Upstream workflow
- [PEP 427](https://peps.python.org/pep-0427/) - Wheel binary package format
- [Docker Bake](docker-bake.hcl) - Existing build configuration
- [CI Workflow Docs](CI_WORKFLOW.md) - SageAttention2 workflow documentation
