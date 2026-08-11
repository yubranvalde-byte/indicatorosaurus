# AlphaQuant AI System Prompt & Engineering Constraints (`ALPHAQUANT_AI_RULES.md`)

> **SYSTEM ROLE & DIRECTIVE**
> You are an expert AI quantitative developer and systems engineer working on **AlphaQuant V3.3**.
> All code generated or modified in this codebase MUST strictly adhere to the mandatory architectural rules defined herein.

---

## 1. C-FFI Memory Struct Alignment Strictness (80-Byte Contract)

- **C Struct Standard (`stock.h` & `financial_engine.c`)**:
  ```c
  #define MAX_TICKER_LEN  8
  #define NUM_INDICATORS 16

  typedef struct {
      char   ticker[MAX_TICKER_LEN];     //  8 bytes — Null-padded ticker symbol string (e.g., "AAPL\0")
      float  indicators[NUM_INDICATORS]; // 64 bytes — Pre-normalized (0-100) float values
      double final_score;                //  8 bytes — Double-precision weighted dot product result
  } Stock;                               // Total: EXACTLY 80 bytes
  ```
- **Python `ctypes` Alignment (`financial_module.py`)**:
  ```python
  class Stock(ctypes.Structure):
      _fields_ = [
          ("ticker",      ctypes.c_char * 8),
          ("indicators",  ctypes.c_float * 16),
          ("final_score", ctypes.c_double),
      ]
  assert ctypes.sizeof(Stock) == 80, f"Stock struct size mismatch: expected 80 bytes, got {ctypes.sizeof(Stock)}"
  ```
- **Mandates**:
  - NEVER alter field ordering or padding in `stock.h` or `financial_module.py` without updating BOTH files simultaneously and asserting `ctypes.sizeof(Stock) == 80` (or updating size assertions accordingly).
  - High-performance C computational kernels (`libsortbuslines.so`) must remain zero-copy and operate on contiguous memory arrays of `Stock`.

---

## 2. NumPy Vectorization Mandate (Zero Python Element-Wise Loops)

- **Core Rule**: All matrix mathematical transformations, statistical normalizations (Goldilocks Gaussian decay, sector-neutral Sigmoid Z-scores), and knock-out gate filtering MUST be fully vectorized via NumPy / Pandas array operations.
- **Prohibited Patterns**:
  - NO scalar `for` loops over rows/elements in Python for factor normalization or scoring.
  - NO list comprehensions for mathematical transformations on indicator matrices.
- **Mandated Operations**:
  - Use `_vectorized_goldilocks` and `_vectorized_sigmoid` operating on $(N, 16)$ float32 NumPy matrices.
  - Use NumPy vectorized conditionals (`np.where`, `np.exp`, `np.clip`, `np.isnan`).

---

## 3. Zero Look-Ahead Bias & Pure Offline Mandate

- **ETL Ingestion (`simfin_etl.py`)**:
  - Fundamental financial statements MUST be joined to price time-series strictly using **Publish Date** constraints:
    $$\text{Publish Date} \le \text{Daily Share Price Date}$$
  - Fundamental data is made visible to the engine ONLY after public filing.
- **Data Persistence & Format**:
  - Historical price time-series MUST be stored in Apache Parquet format (`us-shareprices-daily.parquet`) with `PyArrow` pushdown filters (`Ticker` whitelist + `Date` bounds).
  - Offline baseline fundamental snapshots reside in `historical_fundamental_data.csv`.
  - NO synthetic data backfilling or artificial forward-fill spoofing is permitted. Safe coercion to `0.0` or neutral bounds (`_safe_float`) must be enforced on missing metrics.

---

## 4. Deterministic Script Architecture & Execution Standards

- Heavy autonomous agent multi-step tooling MUST be replaced by deterministic scripts in `/scripts`.
- Core maintenance tasks are driven via standalone Python scripts:
  - `python scripts/compile_c_engine.py`: Compiles C kernel using GCC `-O3 -shared -fPIC` into `libsortbuslines.so` and validates C-FFI struct alignment.
  - `python scripts/run_rolling_backtest.py`: Triggers backtest simulation across historical anchor dates cleanly and outputs key performance metrics.
- 
## 5. Living Documentation Requirement**: Any changes to system architecture, FFI layout, or mathematical rules MUST be reflected in `README.md`.

  - **Living Documentation:** Whenever you modify the codebase, alter the architecture, or adjust the mathematical/algorithmic logic in this project, you MUST proactively update the English `README.md` to reflect these changes.
  - Always maintain a small section specifying the current problems you are trying to fix. When you have fixed them update the README.md with the changes you have made and remove this section.
  - Ensure the documentation always stays synchronized with the current state of the code. Do not use Hebrew for the README; it must remain in English to efficiently pass elaborated descriptive information.
  
## 6. Target Git Repository: Always commit and push code changes in this workspace to the target remote repository: `https://github.com/yubranvalde-byte/finance-app`. Ensure remote URL is configured accordingly.
---
*AlphaQuant V3.3 AI Architectural Specification*
