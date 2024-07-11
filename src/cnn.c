#include "cnn.h"
#include "mat.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

static inline double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

static inline double activate(double x, ActFun act) {
  switch (act) {
    case SIGMOID: return sigmoid(x);
    case RELU: return x > 0.0 ? x : 0.0;
    case TANH: return tanh(x);
    default: assert(0 && "unreachable");
  };
}

static inline double dactf(double a, ActFun act) {
  switch (act) {
    case SIGMOID: return a * (1.0 - a);
    case RELU: return a > 0.0 ? 1.0 : 0.0;
    case TANH: return 1 - a * a;
    default:
      assert(0 && "unreachable");
  }
}

static void activate_Mat2D(Mat2D *m, ActFun act) {
  #pragma omp parallel for
  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      MAT2D_GET((*m), i, j) = activate(MAT2D_GET((*m), i, j), act);
    }
  }
}

static layer_t new_dense_layer(size_t input_size, size_t output_size) {
  layer_t dl = (layer_t) {
    .kind = DENSE,
  };

  dl.dl = (DenseLayer) {
    .a = new_Mat2D(1, output_size),
    .ws = new_Mat2D(input_size, output_size),
    .bias = 0.0,
  };

  return dl;
}

static void destroy_dense_layer(DenseLayer *dl) {
  destroy_Mat2D(&dl->a);
  destroy_Mat2D(&dl->ws);
}

static nn_t nn_copy_structure(const nn_t *nn) {
  assert(nn->layers[0].kind == _INPUT);

  nn_t cpy = (nn_t) {
    .capacity = nn->capacity,
    .layer_count = 1,
    .layers = (layer_t *) malloc(sizeof(layer_t) * nn->capacity),
  };

  cpy.layers[0].kind = _INPUT;
  cpy.layers[0].il.input = nn->layers[0].il.input;

  for (size_t l = 1; l < nn->layer_count; ++l) {
    layer_t layer = nn->layers[l];
    cpy.layers[l].kind = layer.kind;
    switch (layer.kind) {
      case DENSE:
        add_dense_layer(&cpy, layer.dl.ws.rows, layer.dl.ws.cols, layer.act);
        break;
      default:
        assert("unreachable" && 0);
    }
  }

  return cpy;
}

static void dense_layer_learn(DenseLayer *dl, DenseLayer *g, double lr) {
  assert(dl->ws.cols == g->ws.cols && dl->ws.rows == g->ws.rows);

  for (size_t i = 0; i < dl->ws.rows; ++i) {
    for (size_t j = 0; j < dl->ws.cols; ++j) {
      MAT2D_GET(dl->ws, i, j) -= MAT2D_GET(g->ws, i, j) * lr;
    }
  }

  dl->bias -= g->bias * lr;
}

static void nn_learn(nn_t *nn, const nn_t *g, double lr) {
  for (size_t l = 1; l < nn->layer_count; ++l) {
    switch (nn->layers[l].kind) {
      case DENSE:
        dense_layer_learn(&nn->layers[l].dl, &g->layers[l].dl, lr);
        break;
      default:
        assert("unreachable" && 0);
    }
  }
}

nn_t new_nn() {
  nn_t nn = (nn_t) {
    .layer_count = 1,
    .capacity = 10,
    .layers = (layer_t *) malloc(sizeof(layer_t) * 10),
  };

  assert(nn.layers != NULL && "Not enough memory");

  nn.layers[0].kind = _INPUT;
  nn.layers[0].il.input = NULL;

  return nn;
}

void nn_init_random(nn_t *nn, const double min, const double max) {
  const double diff = max - min;

  for (size_t l = 1; l < nn->layer_count; ++l) {
    switch (nn->layers[l].kind) {
      case DENSE:
        random_init_Mat2D(&nn->layers[l].dl.ws, min, max);
        nn->layers[l].dl.bias = (double) random() / (double) RAND_MAX * diff + min;
        break;
      default:
        assert("unreachable" && 0);
    }
  }
}

void nn_init_zero(nn_t *nn) {
  for (size_t l = 1; l < nn->layer_count; ++l) {
    switch (nn->layers[l].kind) {
      case DENSE:
        zero_init_Mat2D(&nn->layers[l].dl.ws);
        zero_init_Mat2D(&nn->layers[l].dl.a);
        nn->layers[l].dl.bias = 0.0;
        break;
      default:
        assert("unreachable" && 0);
    }
  }
}

void nn_destroy(nn_t *nn) {
  for (size_t l = 0; l < nn->layer_count; ++l) {
    switch (nn->layers[l].kind) {
      case DENSE:
        destroy_dense_layer(&nn->layers[l].dl);
        break;
      case _INPUT:
        nn->layers[l].il.input = NULL;
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
  layer_t dl = new_dense_layer(input_size, output_size);
  dl.act = act;

  if (nn->capacity <= nn->layer_count) {
    nn->capacity *= 1.5;
    nn->layers = (layer_t *) realloc(nn->layers, sizeof(layer_t) * nn->capacity);
    assert(nn->layers != NULL && "not enough memory");
  }

  nn->layers[nn->layer_count++] = dl;
}

void nn_forward(nn_t *nn, const Mat2D *input) {
  Mat2D m = *input;
  nn->layers[0].il.input = input;

  for (size_t l = 1; l < nn->layer_count; ++l) {
    layer_t layer = nn->layers[l];
    switch (layer.kind) {
      case DENSE:
        mul_Mat2D(&m, &layer.dl.ws, &layer.dl.a);

        add_scalar_Mat2D(&layer.dl.a, layer.dl.bias);
        activate_Mat2D(&layer.dl.a, layer.act);
        m = layer.dl.a;
        break;
      default:
        assert("unreachable" && 0);
    }
  }
}

inline Mat2D nn_layer_output(const layer_t *l) {
  switch (l->kind) {
    case DENSE:
      return l->dl.a;
    case _INPUT:
      return *l->il.input;
    default:
      assert("unreachable" && 0);
  }
}

Mat2D nn_output(const nn_t *nn) {
  return nn_layer_output(&nn->layers[nn->layer_count - 1]);
}

nn_t nn_backprop(const nn_t *nn, const Mat2D *y) {
  assert(nn->layers[0].kind == _INPUT && nn->layers[0].il.input != NULL);
  nn_t g = nn_copy_structure(nn);
  nn_init_zero(&g);
  layer_t *output_layer = &g.layers[g.layer_count - 1];
  assert(output_layer->kind == DENSE && "output layer must be a dense layer for the moment");
  assert(output_layer->dl.a.cols == y->cols && output_layer->dl.a.rows == y->rows);

  if (output_layer->act == SOFTMAX) {
    assert("softmax is unimplemented" && 0);
  } else {
    Mat2D o = nn_output(nn);
    for (size_t i = 0; i < output_layer->dl.a.cols; ++i) {
      MAT2D_GET(output_layer->dl.a, 0, i) = MAT2D_GET(o, 0, i) - MAT2D_GET((*y), 0, i);
    }
  }

  for (size_t l = g.layer_count - 1; l > 0; --l) {
    if (g.layers[l].kind == DENSE) {
      #pragma omp parallel for shared(g, nn) // TODO: see if this is worth it with the atomics
      for (size_t i = 0; i < g.layers[l].dl.a.cols; ++i) {
        const double de = g.layers[l].dl.a.elems[i];
        const double da = dactf(nn->layers[l].dl.a.elems[i], g.layers[l].act);
        const double delta = de * da;
        #pragma omp atomic
        g.layers[l].dl.bias += delta;

        Mat2D prev_act = nn_layer_output(&g.layers[l-1]);
        for(size_t j = 0; j < prev_act.cols; ++j) {
          const double w = MAT2D_GET(nn->layers[l].dl.ws, j, i);
          const double a = nn_layer_output(&nn->layers[l-1]).elems[j];
          if (l > 1) {
            #pragma omp atomic
            prev_act.elems[j] += w * delta;
          }
          #pragma omp atomic
          MAT2D_GET(g.layers[l].dl.ws, j, i) += a * delta;
        }
      }
    }
  }

  return g;
}

// each row of the train_data is an input
void nn_fit(nn_t *nn, const Mat2D *train_data, const Mat2D *labels, double lr) {
  assert(nn_output(nn).cols == labels->cols && nn_output(nn).rows == labels->rows);

  for (size_t i = 0; i < train_data->rows; ++i) {
    Mat2D x = (Mat2D) {
      .cols = train_data->cols,
      .rows = 1,
      .elems = &train_data->elems[i * train_data->cols],
    };

    Mat2D y = (Mat2D) {
      .cols = labels->cols,
      .rows = 1,
      .elems = &labels->elems[i * labels->cols],
    };

    nn_forward(nn, &x);
    nn_t g = nn_backprop(nn, &y);

    nn_learn(nn, &g, lr);
    nn_destroy(&g);
  }
}
