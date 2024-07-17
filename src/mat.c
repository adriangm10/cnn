#include "mat.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

Mat2D new_Mat2D(const size_t rows, const size_t cols) {
  Mat2D m = (Mat2D) {
    .cols = cols,
    .rows = rows,
    .elems = (double *)malloc(rows * cols * sizeof(double))
  };

  assert(m.elems != NULL && "not enough memory");

  return m;
}

void destroy_Mat2D(Mat2D *m) {
  free(m->elems);
  m->elems = NULL;
  m->cols = 0;
  m->rows = 0;
}

void mul_Mat2D(const Mat2D *m1, const Mat2D *m2, Mat2D *out) {
  assert(m1->cols == m2->rows);
  assert(out->rows == m1->rows && out->cols == m2->cols);

  #pragma omp parallel for shared(m1, m2, out)
  for (size_t i = 0; i < m1->rows; ++i) {
    for (size_t j = 0; j < m2->cols; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < m1->cols; ++k) {
        sum += MAT2D_GET((*m1), i, k) * MAT2D_GET((*m2), k, j);
      }
      MAT2D_GET((*out), i, j) = sum;
    }
  }
}

void random_init_Mat2D(Mat2D *m, const double min, const double max) {
  const double diff = max - min;

  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      MAT2D_GET((*m), i, j) = (double) random() / (double) RAND_MAX * diff + min;
    }
  }
}

void zero_init_Mat2D(Mat2D *m) {
  memset(m->elems, 0, sizeof(double) * m->cols * m->rows);
}

// m += s
void add_scalar_Mat2D(Mat2D *m, const double s) {
  #pragma omp parallel for
  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      MAT2D_GET((*m), i, j) += s;
    }
  }
}

// m1 += m2
void sum_Mat2D(Mat2D *m1, const Mat2D *m2) {
  assert(m1->cols == m2->cols && m1->rows == m2->rows);

  #pragma omp parallel for
  for(size_t i = 0; i < m1->rows; ++i) {
    for (size_t j = 0; j < m1->cols; ++j) {
      MAT2D_GET((*m1), i, j) += MAT2D_GET((*m2), i, j);
    }
  }
}

void print_Mat2D(const Mat2D *m, const char *end) {
  for(size_t i = 0; i < m->rows; ++i) {
    printf("[");
    for (size_t j = 0; j < m->cols; ++j) {
      printf("% .2f", MAT2D_GET((*m), i, j));
    }
    puts(" ]");
  }
  printf("%s", end);
}
