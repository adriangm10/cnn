#include "mat.h"
#include <assert.h>
#include <stdlib.h>
#include <omp.h>

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

Mat2D mul_Mat2D(const Mat2D *m1, const Mat2D *m2) {
  assert(m1->cols == m2->rows);
  Mat2D res = new_Mat2D(m1->rows, m2->cols);

  #pragma omp parallel for shared(res)
  for (size_t i = 0; i < m1->rows; ++i) {
    for (size_t j = 0; j < m2->cols; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < m1->cols; ++k) {
        sum += MAT2D_GET((*m1), i, k) * MAT2D_GET((*m2), k, j);
      }
      MAT2D_GET(res, i, j) = sum;
    }
  }

  return res;
}

Mat2D random_mat(const size_t rows, const size_t cols, const double min, const double max) {
  Mat2D m = new_Mat2D(rows, cols);
  const double diff = max - min;

  #pragma omp parallel for
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT2D_GET(m, i, j) = (double) random() / (double) RAND_MAX * diff + min;
    }
  }

  return m;
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
