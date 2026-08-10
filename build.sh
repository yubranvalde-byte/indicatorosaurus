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

echo "=== 3. Verifying Parquet Data Lake Presence ==="
if [ -f "data/raw_simfin/us-shareprices-daily.parquet" ]; then
    echo "Parquet Data Lake present: $(ls -lh data/raw_simfin/us-shareprices-daily.parquet)"
else
    echo "WARNING: us-shareprices-daily.parquet NOT FOUND in data/raw_simfin/"
    ls -la data/ || true
fi

echo "=== Build Finished Successfully ==="
