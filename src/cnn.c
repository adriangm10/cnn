#include "cnn.h"
#include "mat.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

static double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

static double activate(double x, ActFun act) {
  switch (act) {
    case SIGMOID: return sigmoid(x);
    case RELU: return x > 0.0 ? x : 0.0;
    case TANH: return tanh(x);
    default: assert(0 && "unreachable");
  };
}

static void activate_Mat2D(Mat2D *m, ActFun act) {
  #pragma omp parallel for
  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      MAT2D_GET((*m), i, j) = activate(MAT2D_GET((*m), i, j), act);
    }
  }
}

nn_t new_nn() {
  nn_t nn = (nn_t) {
    .layer_count = 0,
    .capacity = 10,
    .layers = (layer_t *) malloc(sizeof(layer_t) * 10),
  };

  assert(nn.layers != NULL && "Not enough memory");
  return nn;
}

static layer_t new_dense_layer(size_t input_size, size_t output_size, ActFun act) {
  layer_t dl = (layer_t) {
    .kind = DENSE,
  };

  dl.dl = (DenseLayer) {
    .a = new_Mat2D(1, output_size),
    .ws = random_mat(input_size, output_size, -1.0, 1.0),
    .act = act,
    .bias = (double) random() / (double) RAND_MAX * 2.0 - 1.0,
  };

  return dl;
}

static void destroy_dense_layer(DenseLayer *dl) {
  destroy_Mat2D(&dl->a);
  destroy_Mat2D(&dl->ws);
}

void nn_destroy(nn_t *nn) {
  for (size_t l = 0; l < nn->layer_count; ++l) {
    switch (nn->layers[l].kind) {
      case DENSE:
        destroy_dense_layer(&nn->layers[l].dl);
        break;
      default:
        assert(0 && "unreachable");
    }
  }

  free(nn->layers);
  nn->capacity = 0;
  nn->layer_count = 0;
}

void add_dense_layer(nn_t *nn, size_t input_size, size_t output_size, ActFun act) {
  layer_t dl = new_dense_layer(input_size, output_size, act);

  if (nn->capacity <= nn->layer_count) {
    nn->capacity *= 1.5;
    nn->layers = (layer_t *) realloc(nn->layers, sizeof(layer_t) * nn->capacity);
    assert(nn->layers != NULL && "not enough memory");
  }

  nn->layers[nn->layer_count++] = dl;
}

void nn_forward(nn_t *nn, const Mat2D *input) {
  const Mat2D *m = input;

  for (size_t l = 0; l < nn->layer_count; ++l) {
    switch (nn->layers[l].kind) {
      case DENSE:
        nn->layers[l].dl.a = mul_Mat2D(m, &nn->layers[l].dl.ws);
        assert(nn->layers[l].dl.a.rows == 1);

        add_scalar_Mat2D(&nn->layers[l].dl.a, nn->layers[l].dl.bias);
        activate_Mat2D(&nn->layers[l].dl.a, nn->layers[l].dl.act);
        m = &nn->layers[l].dl.a;
        break;
      default:
        assert("unreachable" && 0);
    }
  }
}
