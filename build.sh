#!/usr/bin/env bash
# Render / Cloud Deployment Build Script for AlphaQuant V3.3
# Exit immediately if any command fails
set -o errexit

echo "=== 1. Installing Python Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== 2. Compiling C-FFI Kernel (libsortbuslines.so) ==="
make clean
make

echo "=== Build Finished Successfully ==="
