# SageAttention Build System Refactoring Proposal

## Overview

Currently, SageAttention uses two separate and inconsistent build approaches:
- **Linux builds**: Docker-based builds using `docker-bake.hcl` and `dockerfile.builder`
- **Windows builds**: Native builds using `cibuildwheel` with Windows toolchain

This creates maintenance overhead, inconsistent developer experience, and makes it difficult to add new platforms or architectures.

## Proposed Solution

Standardize on `cibuildwheel` for all platforms to create a unified, maintainable build system that works consistently across Linux, Windows, and macOS while preserving CUDA-specific optimizations.

## Acceptance Criteria

### Functional Requirements
- [ ] **AC1**: Single build command works across all supported platforms (Linux, Windows, macOS)
- [ ] **AC2**: All existing CUDA compute capabilities (8.0, 8.6, 8.9, 9.0, 12.0) are preserved
- [ ] **AC3**: PyTorch version compatibility is maintained (2.6.0, 2.7.0, etc.)
- [ ] **AC4**: Python version support covers 3.9, 3.10, 3.11, 3.12
- [ ] **AC5**: Wheel naming convention remains consistent with current format
- [ ] **AC6**: Build artifacts are identical in functionality to current Docker builds

### Non-Functional Requirements
- [ ] **AC7**: Build time is reduced by at least 20% compared to Docker-based builds
- [ ] **AC8**: Single GitHub Actions workflow replaces both existing workflows
- [ ] **AC9**: Local development builds use same tooling as CI/CD
- [ ] **AC10**: New platforms can be added by modifying configuration files only
- [ ] **AC11**: Build process is documented for developers

### Quality Requirements
- [ ] **AC12**: All existing tests pass with new build artifacts
- [ ] **AC13**: No regression in wheel compatibility or performance
- [ ] **AC14**: Build process is deterministic and reproducible
- [ ] **AC15**: Error messages are clear and actionable

## Task List

### Phase 1: Configuration Setup (Week 1)
- [ ] **T1.1**: Create `cibuildwheel.toml` configuration file
- [ ] **T1.2**: Define build matrix in `build-matrix.yaml`
- [ ] **T1.3**: Update `pyproject.toml` for cross-platform compatibility
- [ ] **T1.4**: Create platform-specific build scripts

### Phase 2: Core Build System Updates (Week 2)
- [ ] **T2.1**: Modify `setup.py` for cross-platform CUDA detection
- [ ] **T2.2**: Update `update_pyproject.py` for multi-platform support
- [ ] **T2.3**: Add platform-specific compiler flag handling
- [ ] **T2.4**: Implement cross-platform CUDA path resolution

### Phase 3: CI/CD Integration (Week 3)
- [ ] **T3.1**: Create unified GitHub Actions workflow
- [ ] **T3.2**: Replace `ci.yml` with new workflow
- [ ] **T3.3**: Replace `build-sageattn.yml` with new workflow
- [ ] **T3.4**: Test workflow on all platforms

### Phase 4: Testing and Validation (Week 4)
- [ ] **T4.1**: Build wheels on all target platforms
- [ ] **T4.2**: Run existing test suite with new wheels
- [ ] **T4.3**: Performance regression testing
- [ ] **T4.4**: Compatibility testing with different PyTorch versions

### Phase 5: Documentation and Cleanup (Week 5)
- [ ] **T5.1**: Update README with new build instructions
- [ ] **T5.2**: Create developer build guide
- [ ] **T5.3**: Document platform-specific considerations
- [ ] **T5.4**: Remove deprecated Docker build files

### Phase 6: Deployment and Monitoring (Week 6)
- [ ] **T6.1**: Deploy new build system to main branch
- [ ] **T6.2**: Monitor build success rates
- [ ] **T6.3**: Collect build performance metrics
- [ ] **T6.4**: Address any post-deployment issues

## Risk Assessment

### High Risk
- **R1**: CUDA compilation differences between platforms causing functional regressions
- **R2**: Build time increases instead of decreases
- **M1**: Extensive testing on all platforms before deployment

### Medium Risk
- **R3**: PyTorch version compatibility issues
- **R4**: Wheel naming convention inconsistencies
- **M2**: Gradual rollout with fallback to existing system

### Low Risk
- **R5**: Documentation gaps
- **R6**: Developer workflow disruption
- **M3**: Comprehensive documentation and training

## Success Metrics

- [ ] **M1**: Build time reduction ≥ 20%
- [ ] **M2**: Single workflow file instead of two
- [ ] **M3**: 100% test pass rate with new wheels
- [ ] **M4**: Zero functional regressions
- [ ] **M5**: Developer satisfaction score ≥ 4.0/5.0

## Dependencies

- **D1**: `cibuildwheel` tool availability and stability
- **D2**: GitHub Actions runner availability for all platforms
- **D3**: CUDA toolkit availability on Windows runners
- **D4**: PyTorch wheel availability for target platforms

## Rollback Plan

If critical issues arise:
1. Revert to previous workflow files
2. Restore Docker-based builds for Linux
3. Maintain Windows `cibuildwheel` builds
4. Investigate and fix issues in separate branch
5. Re-deploy after resolution

## Timeline

- **Total Duration**: 6 weeks
- **Critical Path**: Phases 2-4 (core updates and testing)
- **Milestone 1**: End of Week 2 - Core system ready
- **Milestone 2**: End of Week 4 - All tests passing
- **Milestone 3**: End of Week 6 - Production deployment

## Stakeholders

- **Primary**: SageAttention development team
- **Secondary**: PyTorch ecosystem users
- **Tertiary**: CI/CD infrastructure team
- **Approval Required**: Project maintainers, CI/CD team lead
