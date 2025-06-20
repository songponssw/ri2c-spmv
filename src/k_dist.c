#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conversions.h"
#include "mmio.h"
#include "utils.h"

void print_dist_list(int *data, int N) {
  printf("\"["); // It's already added ","
  for (int i = 0; i < N; i++) {
    printf("%d", data[i]);
    if (i < N - 1) {
      printf(",");
    }
  }
  printf("]\"");
}

void print_dist_dict(int *k, int *v, int N) {
  printf(",\"{"); // It's already added ","
  for (int i = 0; i < N; i++) {
    printf("%d:%d", k[i], v[i]);
    if (i < N - 1) {
      printf(",");
    }
  }
  printf("}\"");
}

int main(int argc, char *argv[]) {
  FILE *f;
  int M, N, nz, inner, inner_max, outer_max;
  int i, *row, *col;
  double *coo_val;

  // Variable for CSR and other formats
  int *row_ptr, *colind;
  double *val;

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

  // int *nnz_row, max_nnz_row, min_nnz_row, *nnz_col;
  // long long int empty_row = 0;
  //
  // nnz_row = calloc(N, sizeof(int));
  // nnz_col = calloc(N, sizeof(int));
  // min_nnz_row = N;
  // max_nnz_row = 0;
  //
  // for (int i = 0; i < nz; i++) {
  //   nnz_row[row[i]]++;
  //   nnz_col[col[i]]++;
  // }
  //

  // DIA distribution
  int *dia_nnz_row = calloc(N, sizeof(int));
  int *dia_nnz_col = calloc(N, sizeof(int));
  int stride, num_diags, *offset;
  double *dia_data;
  coo_csr(nz, N, row, col, coo_val, &row_ptr, &colind, &val);
  csr_dia(row_ptr, colind, val, &offset, &dia_data, N, &num_diags, &stride, nz);

  int *nnz_k;
  nnz_k = calloc(num_diags, sizeof(int));
  int k, istart, iend, index;
  for (int i = 0; i < num_diags; i++) {
    k = offset[i];
    index = 0;
    istart = (0 < -k) ? index = N - stride, -k : 0;
    iend = (N - 1 < N - 1 - k) ? N - 1 : N - 1 - k;
    int k_count = 0;
    for (int n = istart; n <= iend; n++) {
      dia_nnz_row[n]++;     // y[n]
      dia_nnz_col[n + k]++; // x[n+k]
      // Count only nnz
      if (dia_data[(size_t)i * stride + n - index] != 0) {
        k_count++;
      }
    }
    nnz_k[i] = k_count;
  }
  printf("%d",num_diags);
  print_dist_dict(offset, nnz_k, num_diags);

  return 0;
}
