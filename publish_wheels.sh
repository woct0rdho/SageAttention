#!/bin/bash

# Simple script to publish wheels to GitHub Package Registry
# Usage: ./publish_wheels.sh

set -e

# Find wheel files
WHEELS=$(find . -name "*.whl" -type f 2>/dev/null || true)

if [ -z "$WHEELS" ]; then
    echo "No wheel files found"
    exit 1
fi

echo "Found wheels:"
echo "$WHEELS"
echo

# Create a release if none exists
RELEASE_TAG="v2.2.0"
if ! gh release view "$RELEASE_TAG" --repo pixeloven/SageAttention >/dev/null 2>&1; then
    echo "Creating release $RELEASE_TAG..."
    gh release create "$RELEASE_TAG" --title "$RELEASE_TAG" --notes "SageAttention wheel release" --repo pixeloven/SageAttention
fi

# Upload each wheel as a release asset
for wheel in $WHEELS; do
    echo "Uploading $wheel..."
    gh release upload "$RELEASE_TAG" "$wheel" --clobber --repo pixeloven/SageAttention
done

echo "Done!"