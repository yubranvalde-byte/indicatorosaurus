#!/usr/bin/env python3
"""
compile_c_engine.py
===================
Deterministic script to compile the C quantitative engine into a shared library (libsortbuslines.so)
and verify FFI ctypes 80-byte Stock struct alignment.
"""

import os
import sys
import subprocess
import ctypes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_SOURCE = os.path.join(PROJECT_ROOT, "financial_engine.c")
SO_OUTPUT = os.path.join(PROJECT_ROOT, "libsortbuslines.so")


def compile_c_engine():
    print(f"[compile_c_engine] Target C file: {C_SOURCE}")
    print(f"[compile_c_engine] Output shared library: {SO_OUTPUT}")

    if not os.path.exists(C_SOURCE):
        print(f"Error: C source file '{C_SOURCE}' not found.", file=sys.stderr)
        sys.exit(1)

    compilers_to_try = [
        ["gcc", "-O3", "-shared", "-fPIC", C_SOURCE, "-o", SO_OUTPUT],
        ["wsl", "gcc", "-O3", "-shared", "-fPIC", "financial_engine.c", "-o", "libsortbuslines.so"],
    ]

    success = False
    for cmd in compilers_to_try:
        print(f"[compile_c_engine] Trying compiler command: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=PROJECT_ROOT)
            if res.stdout:
                print(res.stdout)
            success = True
            break
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[compile_c_engine] Command '{cmd[0]}' failed or not found: {e}")
            continue

    if not success:
        print("Error: Neither native 'gcc' nor 'wsl gcc' could compile the dynamic library.", file=sys.stderr)
        sys.exit(1)

    # Verify FFI Stock struct alignment
    sys.path.insert(0, PROJECT_ROOT)
    try:
        from financial_module import Stock
        expected_size = 80
        actual_size = ctypes.sizeof(Stock)
        assert actual_size == expected_size, f"Struct alignment mismatch! Expected {expected_size} bytes, got {actual_size} bytes."
        print(f"[compile_c_engine] SUCCESS: Shared library built and Stock struct size verified ({actual_size} bytes).")
    except Exception as err:
        print(f"Verification error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    compile_c_engine()
