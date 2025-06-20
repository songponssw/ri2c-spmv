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

  int *nnz_row, max_nnz_row, min_nnz_row, *nnz_col;
  long long int empty_row = 0;

  nnz_row = calloc(N, sizeof(int));
  nnz_col = calloc(N, sizeof(int));
  min_nnz_row = N;
  max_nnz_row = 0;

  for (int i = 0; i < nz; i++) {
    nnz_row[row[i]]++;
    nnz_col[col[i]]++;
  }

  //  Printing
  //  N,r0,r1,r2,...
  printf("%d,", N);
  print_dist_list(nnz_row,N);
  printf(",");
  print_dist_list(nnz_col,N);
  return 0;
}
