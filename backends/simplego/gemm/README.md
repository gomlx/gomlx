This package implements the GEMM (General Matrix Multiplication) used by the `simplego` backend.


## Performance before for Float32 (Using AVX512):

| Test Name | LHS Dims | RHS Dims | DType | BatchSize | Time/Run | Num Ops | GOps/Sec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `NoBatch-Tiny` | {128, 4} | {4, 1} | Float32 | 1 | 2.74µs | 1,024 | 0.4 |
| `NoBatch-Tiny-Norm` | {128, 4} | {1, 4} | Float32 | 1 | 3.18µs | 1,024 | 0.3 |
| `NoBatch-Small` | {16, 128} | {128, 32} | Float32 | 1 | 25.59µs | 131,072 | 5.1 |
| `NoBatch-Medium` | {128, 128} | {128, 256} | Float32 | 1 | 119.59µs | 8,388,608 | 70.1 |
| `NoBatch-Large` | {1536, 1920} | {1920, 1024} | Float32 | 1 | 17.49ms | 6,039,797,760 | 345.4 |
| `Batched-Large` | {16, 1536, 1920} | {16, 1920, 1024} | Float32 | 16 | 236.51ms | 96,636,764,160 | 408.6 |


