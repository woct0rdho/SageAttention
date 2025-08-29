#!/usr/bin/env python3
"""
SageAttention Build Script
Provides unified interface for both Docker and cibuildwheel builds
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    
    if capture_output:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result
    else:
        result = subprocess.run(cmd, check=check)
        return result


def check_prerequisites():
    """Check if required tools are available"""
    tools = {}
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        tools['docker'] = result.returncode == 0
    except FileNotFoundError:
        tools['docker'] = False
    
    # Check cibuildwheel
    try:
        result = subprocess.run(['cibuildwheel', '--version'], capture_output=True, text=True)
        tools['cibuildwheel'] = result.returncode == 0
    except FileNotFoundError:
        tools['cibuildwheel'] = False
    
    # Check Python
    tools['python'] = True  # We're running Python
    
    return tools


def docker_build(args):
    """Build using Docker"""
    print("🔧 Building with Docker (consistent environment)...")
    
    # Build the Docker image
    docker_cmd = [
        'docker', 'build', 
        '-f', 'dockerfile.builder',
        '-t', 'sageattention-dev',
        '.'
    ]
    
    if args.tag:
        docker_cmd.extend(['-t', args.tag])
    
    run_command(docker_cmd)
    
    if args.run:
        print("🚀 Running interactive shell...")
        run_cmd = [
            'docker', 'run', '-it', '--gpus', 'all',
            'sageattention-dev', '/bin/bash'
        ]
        run_command(run_cmd, check=False)
    
    print("✅ Docker build completed successfully!")


def cibuildwheel_build(args):
    """Build using cibuildwheel"""
    print("⚡ Building with cibuildwheel (fast builds)...")
    
    # Install cibuildwheel if not available
    try:
        subprocess.run(['cibuildwheel', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing cibuildwheel...")
        run_command([sys.executable, '-m', 'pip', 'install', 'cibuildwheel'])
    
    # Set environment variables
    env = os.environ.copy()
    env['TORCH_CUDA_ARCH_LIST'] = args.cuda_arch
    env['CUDA_MINOR_VERSION'] = args.cuda_minor
    env['TORCH_MINOR_VERSION'] = args.torch_minor
    env['TORCH_PATCH_VERSION'] = args.torch_patch
    
    # Build command
    build_cmd = [
        'cibuildwheel',
        '--platform', args.platform,
        '--output-dir', 'wheelhouse'
    ]
    
    if args.only:
        build_cmd.extend(['--only', args.only])
    
    # Run the build
    print(f"Building for platform: {args.platform}")
    print(f"CUDA architectures: {args.cuda_arch}")
    print(f"PyTorch version: 2.{args.torch_minor}.{args.torch_patch}")
    
    run_command(build_cmd, env=env)
    
    print("✅ cibuildwheel build completed successfully!")
    
    # List generated wheels
    wheelhouse = Path('wheelhouse')
    if wheelhouse.exists():
        wheels = list(wheelhouse.glob('*.whl'))
        if wheels:
            print(f"\n📦 Generated wheels ({len(wheels)}):")
            for wheel in wheels:
                print(f"  - {wheel.name}")


def test_wheels(args):
    """Test wheels using Docker"""
    print("🧪 Testing wheels with Docker...")
    
    # Check if wheels exist
    wheelhouse = Path('wheelhouse')
    if not wheelhouse.exists() or not list(wheelhouse.glob('*.whl')):
        print("❌ No wheels found in wheelhouse/ directory")
        print("   Run a build first: python build.py cibuildwheel")
        return
    
    # Test using Docker bake
    test_cmd = [
        'docker', 'buildx', 'bake',
        '--file', 'docker-bake.hcl',
        'test'
    ]
    
    # Set environment variables for Docker bake
    env = os.environ.copy()
    env['CUDA_VERSION'] = f"12.{args.cuda_minor}.1"
    env['PYTHON_VERSION'] = "3.12"
    env['TORCH_CUDA_ARCH_LIST'] = args.cuda_arch.replace(' ', ';')
    env['TORCH_MINOR_VERSION'] = args.torch_minor
    env['TORCH_PATCH_VERSION'] = args.torch_patch
    env['CUDA_MINOR_VERSION'] = args.cuda_minor
    
    run_command(test_cmd, env=env)
    
    print("✅ Wheel testing completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="SageAttention Build Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with Docker (development)
  python build.py docker
  
  # Build with cibuildwheel (production)
  python build.py cibuildwheel
  
  # Build for specific platform
  python build.py cibuildwheel --platform linux
  
  # Test wheels
  python build.py test
  
  # Build and run Docker container
  python build.py docker --run
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Build command')
    
    # Docker build parser
    docker_parser = subparsers.add_parser('docker', help='Build using Docker')
    docker_parser.add_argument('--tag', '-t', help='Additional Docker tag')
    docker_parser.add_argument('--run', '-r', action='store_true', help='Run container after build')
    
    # cibuildwheel build parser
    cibuildwheel_parser = subparsers.add_parser('cibuildwheel', help='Build using cibuildwheel')
    cibuildwheel_parser.add_argument('--platform', '-p', default='auto', 
                                   choices=['auto', 'linux', 'windows'],
                                   help='Target platform')
    cibuildwheel_parser.add_argument('--only', help='Build only specific wheel tags')
    cibuildwheel_parser.add_argument('--cuda-arch', default='8.0 8.6 8.9 9.0 12.0',
                                   help='CUDA architectures to target')
    cibuildwheel_parser.add_argument('--cuda-minor', default='13',
                                   help='CUDA minor version (9 for 12.9, 13 for 13.0)')
    cibuildwheel_parser.add_argument('--torch-minor', default='8',
                                   help='PyTorch minor version')
    cibuildwheel_parser.add_argument('--torch-patch', default='0',
                                   help='PyTorch patch version')
    
    # Test parser
    test_parser = subparsers.add_parser('test', help='Test wheels using Docker')
    test_parser.add_argument('--cuda-arch', default='8.0 8.6 8.9 9.0 12.0',
                            help='CUDA architectures to target')
    test_parser.add_argument('--cuda-minor', default='13',
                            help='CUDA minor version (9 for 12.9, 13 for 13.0)')
    test_parser.add_argument('--torch-minor', default='8',
                            help='PyTorch minor version')
    test_parser.add_argument('--torch-patch', default='0',
                            help='PyTorch patch version')
    
    # Check prerequisites
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check prerequisites
    tools = check_prerequisites()
    print("🔍 Checking prerequisites...")
    for tool, available in tools.items():
        status = "✅" if available else "❌"
        print(f"  {status} {tool}")
    
    if not tools['docker'] and args.command in ['docker', 'test']:
        print("❌ Docker is required for this command")
        print("   Install Docker: https://docs.docker.com/get-docker/")
        return
    
    # Execute command
    if args.command == 'docker':
        docker_build(args)
    elif args.command == 'cibuildwheel':
        cibuildwheel_build(args)
    elif args.command == 'test':
        test_wheels(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()


if __name__ == '__main__':
    main()
