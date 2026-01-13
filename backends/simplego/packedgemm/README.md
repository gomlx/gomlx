This package implements the GEMM (General Matrix Multiplication) used by the `simplego` backend.


## Performance before for Float32 (Using AVX512):

| Test Name | LHS Dims | RHS Dims | DType | BatchSize | Time/Run | Num Ops | GOps/Sec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `NoBatch-Medium` | {128, 128} | {128, 256} | Float32 | 1 | 119.59µs | 8,388,608 | 70.1 |
| `NoBatch-Large` | {1536, 1920} | {1920, 1024} | Float32 | 1 | 17.49ms | 6,039,797,760 | 345.4 |
| `Batched-Large` | {16, 1536, 1920} | {16, 1920, 1024} | Float32 | 16 | 236.51ms | 96,636,764,160 | 408.6 |

## Performance after for Float32 (Using AVX512):

| Test Name | LHS Dims | RHS Dims | DType | BatchSize | Time/Run | Num Ops | GOps/Sec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `NoBatch-Medium` | {128, 128} | {128, 256} | Float32 | 1 | 8.97µs | 8,388,608 | 934.7 |
| `NoBatch-Large` | {1536, 1920} | {1920, 1024} | Float32 | 1 | 2.98ms | 6,039,797,760 | 2025.1 |
| `Batched-Large` | {16, 1536, 1920} | {16, 1920, 1024} | Float32 | 16 | 47.39ms | 96,636,764,160 | 2039.2 |
