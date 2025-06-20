#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conversions.h"
#include "mmio.h"
#include "spmv.h"
#include "utils.h"

#include <sys/time.h>
#include <time.h>

#define OUTER_MAX 10

int main(int argc, char *argv[]) {
  // Variables For read matrix market
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
  char *storage_format = argv[2];
  if (strcmp(storage_format, "coo") == 0) {
    option = 1;
  } else if (strcmp(storage_format, "csr") == 0) {
    option = 2;
  } else if (strcmp(storage_format, "dia") == 0) {
    option = 3;
  } else if (strcmp(storage_format, "ell") == 0) {
    option = 4;
  }

  switch (option) {
  case 1: // COO
  {
    for (i = 0; i < OUTER_MAX; i++) {
      // Warmup
      for (int k = 0; k < 100; k++) {
        spmv_coo(row, col, coo_val, nz, N, x, y);
      }
      zero_arr(N, y);

      // Measure SpMV function
      gettimeofday(&sstart_day, NULL);
      for (inner = 0; inner < inner_max; inner++) {
        spmv_coo(row, col, coo_val, nz, N, x, y);
      }
      gettimeofday(&sstop_day, NULL);

      // Collect exectime in usec
      t_val = ((sstop_day.tv_sec * 1e6 + sstop_day.tv_usec) -
               (sstart_day.tv_sec * 1e6 + sstart_day.tv_usec));
      exec_arr[i] = (double)(t_val);
    }
    break;
  }
  case 2: // CSR
  {
    // Conversion form COO
    gettimeofday(&sstart_day, NULL);
    coo_csr(nz, N, row, col, coo_val, &row_ptr, &colind, &val);
    gettimeofday(&sstop_day, NULL);
    csr_conv = (double)((sstop_day.tv_sec * 1e6) + sstop_day.tv_usec) -
               ((sstart_day.tv_sec * 1e6) + sstart_day.tv_usec);

    for (i = 0; i < OUTER_MAX; i++) {
      for (int k = 0; k < 100; k++) {
        spmv_csr(row_ptr, colind, val, N, x, y);
      }
      zero_arr(N, y);

      gettimeofday(&sstart_day, NULL);
      for (inner = 0; inner < inner_max; inner++) {
        spmv_csr(row_ptr, colind, val, N, x, y);
      }
      gettimeofday(&sstop_day, NULL);

      t_val = ((sstop_day.tv_sec * 1e6 + sstop_day.tv_usec) -
               (sstart_day.tv_sec * 1e6 + sstart_day.tv_usec));
      exec_arr[i] = (double)(t_val);
    }
    break;
  }
  case 3: // DIA
  {
    int stride, num_diags, *offset;
    double *dia_data;

    // Convert to CSR.
    gettimeofday(&sstart_day, NULL);
    coo_csr(nz, N, row, col, coo_val, &row_ptr, &colind, &val);
    gettimeofday(&sstop_day, NULL);
    csr_conv = (double)((sstop_day.tv_sec * 1e6) + sstop_day.tv_usec) -
               ((sstart_day.tv_sec * 1e6) + sstart_day.tv_usec);

    // Convert to target format.
    gettimeofday(&sstart_day, NULL);
    csr_dia(row_ptr, colind, val, &offset, &dia_data, N, &num_diags, &stride,
            nz);
    gettimeofday(&sstop_day, NULL);
    target_conv = ((sstop_day.tv_sec * 1e6 + sstop_day.tv_usec) -
                   (sstart_day.tv_sec * 1e6 + sstart_day.tv_usec));

    for (i = 0; i < OUTER_MAX; i++) {
      for (int k = 0; k < 100; k++) {
        spmv_dia(offset, dia_data, N, num_diags, stride, x, y);
      }
      zero_arr(N, y);

      gettimeofday(&sstart_day, NULL);
      for (inner = 0; inner < inner_max; inner++) {
        spmv_dia(offset, dia_data, N, num_diags, stride, x, y);
      }
      gettimeofday(&sstop_day, NULL);

      t_val = ((sstop_day.tv_sec * 1e6 + sstop_day.tv_usec) -
               (sstart_day.tv_sec * 1e6 + sstart_day.tv_usec));
      exec_arr[i] = (double)(t_val);
    }
    break;
  }
  case 4: // ELL format
  {
    int num_cols, *indices;
    double *ell_data;

    // Convert to target CSR.
    gettimeofday(&sstart_day, NULL);
    coo_csr(nz, N, row, col, coo_val, &row_ptr, &colind, &val);
    gettimeofday(&sstop_day, NULL);
    csr_conv = (double)((sstop_day.tv_sec * 1e6) + sstop_day.tv_usec) -
               ((sstart_day.tv_sec * 1e6) + sstart_day.tv_usec);

    // Convert to target format.
    gettimeofday(&sstart_day, NULL);
    csr_ell(row_ptr, colind, val, &indices, &ell_data, N, &num_cols, nz);
    gettimeofday(&sstop_day, NULL);
    target_conv = ((sstop_day.tv_sec * 1e6 + sstop_day.tv_usec) -
                   (sstart_day.tv_sec * 1e6 + sstart_day.tv_usec));

    for (i = 0; i < OUTER_MAX; i++) {
      for (int k = 0; k < 100; k++) {
        spmv_ell(indices, ell_data, N, num_cols, x, y);
      }
      zero_arr(N, y);

      gettimeofday(&sstart_day, NULL);
      for (inner = 0; inner < inner_max; inner++) {
        spmv_ell(indices, ell_data, N, num_cols, x, y);
      }
      gettimeofday(&sstop_day, NULL);

      t_val = ((sstop_day.tv_sec * 1e6 + sstop_day.tv_usec) -
               (sstart_day.tv_sec * 1e6 + sstart_day.tv_usec));
      exec_arr[i] = (double)(t_val);
    }
    break;
  }
  default:
    printf("Default case\n");
    break;
  }

/**
 *
 * Print Result of an experiment
 * Format: r0,r1,..,csr_conv,tar_conv,inner,outer
 **/
#ifndef PRINT
  for (int k = 0; k < OUTER_MAX; k++) {
    printf("%g,", (double)(exec_arr[k])); // in usec
  }
  printf("%g,", (double)(csr_conv / 1e6));    // in sec
  printf("%g,", (double)(target_conv / 1e6)); // in sec
  printf("%g,", (double)inner_max);
  printf("%g", (double)OUTER_MAX);
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
