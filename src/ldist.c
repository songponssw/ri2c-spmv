// This code measure spme_omp in 4 formats. Similar to exectime,
// but having different option and spmv kernel
#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #include "config.h"
#include "conversions.h"
#include "mmio.h"
#include "spmv_omp.h" // Link to each spmv omp kernel
#include "utils.h"

#include <omp.h>
#include <sys/time.h>
#include <time.h>

#define OUTER_MAX 10

int main(int argc, char *argv[]) {
  // Variables For matrix market reading
  FILE *f;
  int M, N, nz, inner, inner_max, outer_max;
  int i, *row, *col;
  double *coo_val;

  /*
   *
   * Read Input matrix
   *
   */
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [mm filename]\n", argv[0]);
    exit(1);
  } else {
    if ((f = fopen(argv[1], "r")) == NULL) {
      fprintf(stderr, "File Not Found\n");
      exit(1);
    }
  }
  get_matrix_market(f, &M, &N, &nz, &row, &col, &coo_val);

  // Time constants
  struct timeval sstart_day, sstop_day;
  time_t t_val = 0;
  time_t csr_conv = 0;
  time_t target_conv = 0;
  double exec_arr[OUTER_MAX];

  // Variable for CSR and other formats
  int *row_ptr, *colind;
  double *val;
  int s, l; // COO gride size, printing at the end

  // Create Vector
  double *x, *y;
  x = (double *)malloc(N * sizeof(double));
  y = (double *)calloc(N, sizeof(double));
  init_arr(N, x);

  // Inner loop configuration
  inner_max = 2000;
  if (nz > 1000000) {
    inner_max = 1;
  } else if (nz > 100000) {
    inner_max = 10;
  } else if (nz > 50000) {
    inner_max = 20;
  } else if (nz > 20000) {
    inner_max = 50;
  } else if (nz > 5000) {
    inner_max = 200;
  } else if (nz > 500) {
    inner_max = 2000;
  } else {
    fprintf(stderr, "matrices nz < 500\n");
    return 0;
  }

#ifdef PRINT
  inner_max = 1;
#endif

  /*
   *
   * Control Flow: Selecting target format
   *
   */
  int option = 0;
  char *storageFormat = argv[2];
  if (strcmp(storageFormat, "coo") == 0) {
    option = 5;
  } else if (strcmp(storageFormat, "csr") == 0) {
    option = 6;
  } else if (strcmp(storageFormat, "dia") == 0) {
    option = 7;
  } else if (strcmp(storageFormat, "ell") == 0) {
    option = 8;
  }

  // Store data for ldist
  int num_threads = omp_get_max_threads();
  long long int ldist[num_threads];

  for (int i = 0; i < num_threads; i++) {
    ldist[i] = 0;
  }

  switch (option) {

  case 5: // COO
  {
    int num_threads = omp_get_max_threads();
    double **y_local;
    y_local = (double **)calloc(num_threads, sizeof(double *));

    for (int t = 0; t < num_threads; t++) {
      y_local[t] = (double *)calloc(N, sizeof(double));
    }


    spmv_coo_omp_ldist(row, col, coo_val, nz, N, x, y, y_local, ldist);
    break;
  }

  case 6: // CSR
  {
    coo_csr(nz, N, row, col, coo_val, &row_ptr, &colind, &val);
    spmv_csr_omp_ldist(row_ptr, colind, val, N, x, y, ldist);
    break;
  }
  case 7: // DIA data 1-dim
  {
    int stride, num_diags, *offset;
    double *dia_data;

    coo_csr(nz, N, row, col, coo_val, &row_ptr, &colind, &val);
    csr_dia(row_ptr, colind, val, &offset, &dia_data, N, &num_diags, &stride,
            nz);
    spmv_dia_omp_ldist(offset, dia_data, N, num_diags, stride, x, y, ldist);

    break;
  }

  case 8: // ELL format
  {
    int num_cols, *indices;
    double *ell_data;

    coo_csr(nz, N, row, col, coo_val, &row_ptr, &colind, &val);
    csr_ell(row_ptr, colind, val, &indices, &ell_data, N, &num_cols, nz);
    spmv_ell_omp_ldist(indices, ell_data, N, num_cols, x, y, ldist);
    break;
  }

  default:
    printf("Default case\n");
    break;
  }

/**
 *
 * Print Result of an experiment
 * Format: spmv exec[1](usec), spmv_exec[2], ... , csr_conv(sec),
 *target_conv(sec), inner, outer
 **/
#ifndef PRINT
  for (i = 0; i < num_threads; i++) {
    printf("%lld%s", ldist[i], (i < num_threads - 1) ? "," : "");
  }
#endif

/**
 *
 * Print Y Result for correctness
 *
 **/
#ifdef PRINT
  for (i = 0; i < N; i++) {
    printf("%f\t%f\n", y[i], x[i]);
  }
#endif

  return 0;
}
