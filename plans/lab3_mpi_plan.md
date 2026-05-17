# Lab 3: Parallel Matrix Multiplication with MPI — File Creation Plan

## Files to Create/Update

### 1. `generate_matrices.py` — ✅ Already exists (no changes needed)
- Functionally identical to the provided version.

### 2. `verify_result.py` — ✅ Already exists (no changes needed)
- Functionally identical to the provided version.

### 3. `matrix_multiply_mpi.cpp` — ⚠️ Created but EMPTY, needs content
- MPI parallel matrix multiplication implementation
- Uses `MPI_Scatterv` / `MPI_Gatherv` for row distribution
- Uses `MPI_Bcast` to broadcast matrix B to all processes
- Measures computation time with `MPI_Wtime` around the compute kernel only
- Writes result to `data/C.txt` with 15-digit precision
- Outputs matrix size, process count, elapsed time, and GFLOPS

### 4. `run_mpi_experiments.py` — ❌ Does not exist yet, needs creation
- Automated experiment runner and visualization script
- Compiles the MPI program with `mpicxx -O2`
- Runs experiments for sizes: 200, 400, 800, 1200, 1600, 2000
- Runs with process counts: 1, 2, 4, 8
- Takes best of 3 attempts for each configuration
- Verifies results against NumPy
- Saves `mpi_performance.csv` and `mpi_performance.png`

## Action Required
Switch to **Code mode** to write the contents of:
1. `matrix_multiply_mpi.cpp` (currently empty)
2. `run_mpi_experiments.py` (needs to be created)
