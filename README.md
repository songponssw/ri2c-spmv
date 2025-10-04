# Sparse Matrix-Vector Multiplication Analysis Framework


# Overview
This repository provides the experiment source code for [Revisiting Matrix Structure Impact on Storage Format SpMV Performance](#). The work evaluates SpMV execution time, matrix features, and matrix layouts, and uses these results for performance analysis. The pipeline begins with input matrices from [SuiteSparse](https://sparse.tamu.edu/) in **Matrix Market format**. The outputs from measures are in CSV format, which are later summarized into a `pickle` format for plotting and analysis.


```
[ Input matrix (Matrix market) ]
        |
        v
[ Metric Collection ] -> binaries -> [ output/{DATASET}/*.csv ]
        |
        v
[ Preprocessing ] -> [ *.pkl ] -> plotting & analysis
```
---
## Collect metrics and data
### Extract Files from Matrix Market
We use the following directory structure to organize matrix datasets. Matrices downloaded from SuiteSparse must include both the *Group* and *Name* properties.  

For example, the [Pres_Poisson](https://sparse.tamu.edu/ACUSIM/Pres_Poisson) matrix belongs to the **ACUSIM** group and has the **Pres_Poisson** name.  

After downloading, the input matrix should be placed like this:  
```bash
/ACUSIM/Pres_Poisson/Pres_Poisson.mtx
```

### File Structure
The `src/` directory contains the code for measuring SpMV.  The measured functions are implemented separately in `src/spmv_module/`.  Each experiment has its own `.c` file with the measurement code.  
```bash
├── scripts
├── src
│   ├── include/       
│   ├── spmv_kernel/       # target function Implementation 
│   ├── conversions.c
│   ├── exectime.c         # Measure execution time in serial impl
│   ├── exectime_omp.c     # Measure execution time in parallel impl
│   ├── features.c         # Measure matrix feature
│   ├── k_dist.c           # Measure NNZ distribution in diagonals
│   ├── ldist.c            # collect NNZ in each thread
│   ├── Makefile
│   ├── mmio.c
│   ├── row_dist.c
│   └── utils.c
└── src-python             # Details on preprocessing data section
```

### Usage 
```bash
make [exectime,exectime_omp,k_dist,...]
./bin/run_exectime INPUT_PATH coo 0  # COO serial impl.
./bin/run_exectime INPUT_PATH coo 8  # COO serial impl.
```

### Usage (scripts)
The `scripts/` directory provides helper scripts to run measurements on multiple matrices (datasets). Dataset names are defined in `scripts/mtx-name/{DATASET}.name`, which store the paths to matrices. These paths are passed as arguments when running experiments.  

For example, using the dataset name `test2`:  
```bash
make exectime test2
```


---
## Preprocessing Data
The preprocessing step converts the raw measured metrics into `.pkl` files. It also performs additional tasks such as classifying matrices by layout, collecting extra feature metrics, and averaging execution time.  

## File Structure
```bash
├── class_feature.py            # Collect class matrix features 
├── Makefile
├── merge_pkl.py
├── process_features.py         # convert output/{DATASET}/features.csv
├── process_ft.py               # Convert format-thread execution time
├── process_k_dist.py           
├── process_perf_metric.py      # Calculate Speedup, parallel efficiency, ...
├── process_submatrices.py       
└── utils.py
```

## Usage
```bash
cd src-python
make all test2
```


## License
This project is licensed under the [MIT License](./LICENSE).
