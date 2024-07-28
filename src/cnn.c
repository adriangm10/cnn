#include "cnn.h"
#include "mat.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

static inline double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

static double vec_max(const double *x, size_t x_size) {
  double max = x[0];
  for (size_t i = 1; i < x_size; ++i) {
    max = max > x[i] ? max : x[i];
  }
  return max;
}

static void softmax(double *x, size_t x_size) {
  double sum = 0.0;
  double max = vec_max(x, x_size);

  for (size_t i = 0; i < x_size; ++i) {
    x[i] -= max;
    sum += exp(x[i]);
  }

  for (size_t i = 0; i < x_size; ++i) {
    x[i] = exp(x[i]) / sum;
  }
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
    case SOFTMAX: return 1; // suposing it's used in last layer, the derivative with cross entropy loss would be just a - y
    default: assert(0 && "unreachable");
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

static void activate_col(Mat2D *col, ActFun act) {
  assert(col->cols == 1);

  if (act == SOFTMAX) {
    softmax(col->elems, col->rows);
    return;
  }

  #pragma omp parallel for
  for (size_t i = 0; i < col->rows; ++i) {
    col->elems[i] = activate(col->elems[i], act);
  }
}

static layer_t new_dense_layer(size_t size, ActFun act) {
  layer_t dl = (layer_t) {
    .kind = DENSE,
    .act = act,
  };

  dl.dl = (DenseLayer) {
    .a = new_Mat2D(size, 1),
    .bias = 0.0,
  };

  return dl;
}

static void destroy_dense_layer(DenseLayer *dl) {
  destroy_Mat2D(&dl->a);
  destroy_Mat2D(&dl->ws);
}

static layer_t new_conv2d_layer(size_t kernel_count, size_t kernel_size, size_t channels, int padding, int stride, ActFun act) {
  Conv2dLayer cl = {
    .kernel_count = kernel_count,
    .channels = channels,
    .padding = padding,
    .stride = stride,
    .kernels = (Mat2D *) malloc(sizeof(Mat2D) * kernel_count * channels),
    .a = NULL,
  };

  return (layer_t) {
    .kind = CONV2D,
    .act = act,
    .cl = cl,
  };
}

static void destroy_conv2d_layer(Conv2dLayer *l) {
  for (size_t k = 0; k < l->kernel_count * l->channels; ++k) {
    destroy_Mat2D(&l->kernels[k]);
  }
  free(l->kernels);
  l->kernels = NULL;

  if (!l->a) return;
  for (size_t a = 0; a < l->kernel_count; ++a) {
    destroy_Mat2D(&l->a[a]);
  }
  free(l->a);
  l->a = NULL;
}

static layer_t new_pooling_layer(size_t pool_size, enum layer_kind kind) {
  PoolingLayer pl = {
    .channels = -1,
    .pool_size = pool_size,
    .a = NULL,
  };

  return (layer_t) {
    .kind = kind,
    .pl = pl,
  };
}

static void destroy_pooling_layer(PoolingLayer *l) {
  for (size_t a = 0; a < l->channels; ++a) {
    destroy_Mat2D(&l->a[a]);
  }
  free(l->a);
  l->a = NULL;
  l->channels = -1;
};

static layer_t new_flatten_layer() {
  return (layer_t) {
    .kind = FLATTEN,
    .fl = (FlattenLayer) { .a.elems = NULL },
  };
}

static void destroy_flatten_layer(FlattenLayer *l) {
  if (l->a.elems) destroy_Mat2D(&l->a);
}

static void append_layer(nn_t *nn, layer_t l) {
  if (nn->capacity <= nn->layer_count) {
    nn->capacity *= 1.5;
    nn->layers = (layer_t *) realloc(nn->layers, sizeof(layer_t) * nn->capacity);
    assert(nn->layers != NULL && "not enough memory");
  }

  nn->layers[nn->layer_count++] = l;
}

static nn_t nn_copy_structure(const nn_t *nn) {
  assert(nn->layers[0].kind == _INPUT);

  nn_t cpy = (nn_t) {
    .capacity = nn->capacity,
    .layer_count = nn->layer_count,
    .layers = (layer_t *) malloc(sizeof(layer_t) * nn->capacity),
  };

  cpy.layers[0].kind = _INPUT;
  cpy.layers[0].il.input = nn->layers[0].il.input;

  for (size_t l = 1; l < nn->layer_count; ++l) {
    layer_t layer = nn->layers[l];
    cpy.layers[l].kind = layer.kind;
    switch (layer.kind) {
      case DENSE:
        cpy.layers[l].dl.a = new_Mat2D(layer.dl.a.rows, layer.dl.a.cols);
        cpy.layers[l].dl.ws = new_Mat2D(layer.dl.ws.rows, layer.dl.ws.cols);
        cpy.layers[l].dl.bias = 0.0;
        cpy.layers[l].act = layer.act;
        break;
      default:
        assert("unreachable" && 0);
    }
  }

  return cpy;
}

static void dense_layer_learn(DenseLayer *dl, DenseLayer *g, size_t batch_size, double lr) {
  assert(dl->ws.cols == g->ws.cols && dl->ws.rows == g->ws.rows);

  for (size_t i = 0; i < dl->ws.rows; ++i) {
    for (size_t j = 0; j < dl->ws.cols; ++j) {
      MAT2D_GET(dl->ws, i, j) -= MAT2D_GET(g->ws, i, j) * lr / batch_size;
    }
  }

  dl->bias -= g->bias * lr / batch_size;
}

static void nn_learn(nn_t *nn, const nn_t *g, size_t batch_size, double lr) {
  for (size_t l = 1; l < nn->layer_count; ++l) {
    switch (nn->layers[l].kind) {
      case DENSE:
        dense_layer_learn(&nn->layers[l].dl, &g->layers[l].dl, batch_size, lr);
        break;
      default:
        assert("unreachable" && 0);
    }
  }
}

// g1 += g2
static void nn_add_gradient(nn_t *g1, const nn_t *g2) {
  assert(g1->layer_count == g2->layer_count);

  for(size_t l = 1; l < g1->layer_count; ++l) {
    assert(g1->layers[l].kind == g2->layers[l].kind);

    switch (g1->layers[l].kind) {
      case DENSE:
        sum_Mat2D(&g1->layers[l].dl.ws, &g2->layers[l].dl.ws);
        g1->layers[l].dl.bias += g2->layers[l].dl.bias;
        break;
      default:
        assert("unreachable" && 0);
    }
  }
}

nn_t new_nn(size_t height, size_t width, size_t channels) {
  nn_t nn = (nn_t) {
    .layer_count = 1,
    .capacity = 10,
    .layers = (layer_t *) malloc(sizeof(layer_t) * 10),
  };

  assert(nn.layers != NULL && "Not enough memory");

  nn.layers[0].kind = _INPUT;
  nn.layers[0].il.input = NULL;
  nn.layers[0].il.width = width;
  nn.layers[0].il.height = height;
  nn.layers[0].il.channels = channels;

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

void nn_add_dense_layer(nn_t *nn, size_t size, ActFun act) {
  layer_t dl = new_dense_layer(size, act);

  append_layer(nn, dl);
}

void nn_add_conv2d_layer(nn_t *nn, size_t kernel_count, size_t kernel_size, size_t channels, int padding, int stride, ActFun act) {
  layer_t conv = new_conv2d_layer(kernel_count, kernel_size, channels, padding, stride, act);

  append_layer(nn, conv);
}

void nn_add_max_pooling_layer(nn_t *nn, size_t pool_size) {
  layer_t l = new_pooling_layer(pool_size, MAX_POOL);

  append_layer(nn, l);
}

void nn_add_avg_pooling_layer(nn_t *nn, size_t pool_size) {
  layer_t l = new_pooling_layer(pool_size, AVG_POOL);

  append_layer(nn, l);
}

void nn_add_flatten_layer(nn_t *nn) {
  layer_t l = new_flatten_layer();

  append_layer(nn, l);
}

void nn_compile(nn_t *nn) {
  size_t width, height, channels, flatten_size;

  for (size_t l = 0; l < nn->layer_count; ++l) {
    layer_t *layer = &nn->layers[l];
    switch (layer->kind) {
      case _INPUT:
        width = layer->il.width;
        height = layer->il.height;
        channels = layer->il.channels;
        if (width == 1) flatten_size = width * height;
        break;
      case DENSE:
        layer->dl.ws = new_Mat2D(flatten_size, layer->dl.a.rows);
        flatten_size = layer->dl.a.rows;
        break;
      case CONV2D:
        height = (height - layer->cl.kernels[0].rows + 2 * layer->cl.padding) / layer->cl.stride + 1;
        width = (width - layer->cl.kernels[0].cols + 2 * layer->cl.padding) / layer->cl.stride + 1;

        for (size_t i = 0; i < layer->cl.kernel_count; ++i) {
          layer->cl.a[i] = new_Mat2D(height, width);
        }
        channels = layer->cl.kernel_count;
        break;
      case MAX_POOL:
      case AVG_POOL:
        layer->pl.channels = channels;
        width = width / layer->pl.pool_size;
        height = height / layer->pl.pool_size;

        for (size_t c = 0; c < channels; ++c) {
          layer->pl.a[c] = new_Mat2D(height, width);
        }
        break;
      case FLATTEN:
        flatten_size = height * width * channels;
        layer->fl.a = new_Mat2D(flatten_size, 1);
        break;
      default: assert(0 && "unreachable");
    }
  }
}

void nn_forward(nn_t *nn, const Mat2D *input) {
  assert(nn->layers[0].kind == _INPUT);

  Mat2D m = *input;
  Mat2D ws_t;
  nn->layers[0].il.input = input;

  for (size_t l = 1; l < nn->layer_count; ++l) {
    layer_t layer = nn->layers[l];
    switch (layer.kind) {
      case DENSE:
        ws_t = transpose_Mat2D(&layer.dl.ws);
        Mat2D_col_mul(&ws_t, &m, &layer.dl.a);
        destroy_Mat2D(&ws_t);

        add_column_scalar(&layer.dl.a, layer.dl.bias);
        activate_col(&layer.dl.a, layer.act);
        m = layer.dl.a;
        break;
      case CONV2D:
        break;
      case MAX_POOL:
        break;
      case AVG_POOL:
        break;
      case FLATTEN:
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
  Mat2D o = nn_output(nn);
  assert(nn->layers[0].kind == _INPUT && nn->layers[0].il.input != NULL);
  assert(nn->layers[nn->layer_count - 1].kind == DENSE && "output layer must be a dense layer");
  assert(o.cols == y->cols && o.rows == y->rows);

  nn_t g = nn_copy_structure(nn);
  nn_init_zero(&g);
  Mat2D g_o = nn_output(&g);

  for (size_t i = 0; i < g_o.rows; ++i) {
    g_o.elems[i] = o.elems[i] - y->elems[i];
  }

  for (size_t l = g.layer_count - 1; l > 0; --l) {
    if (g.layers[l].kind == DENSE) {
      #pragma omp parallel for shared(g, nn, l) // TODO: see if this is worth with the atomics
      for (size_t i = 0; i < g.layers[l].dl.a.rows; ++i) {
        const double de = g.layers[l].dl.a.elems[i];
        const double da = dactf(nn->layers[l].dl.a.elems[i], g.layers[l].act);
        const double delta = de * da;
        #pragma omp atomic
        g.layers[l].dl.bias += delta;

        Mat2D prev_act = nn_layer_output(&g.layers[l-1]);
        for(size_t j = 0; j < prev_act.rows; ++j) {
          const double w = MAT2D_GET(nn->layers[l].dl.ws, j, i);
          const double a = nn_layer_output(&nn->layers[l-1]).elems[j];
          if (l > 1) {
            #pragma omp atomic
            prev_act.elems[j] += w * delta;
          }
          MAT2D_GET(g.layers[l].dl.ws, j, i) += a * delta;
        }
      }
    }
  }

  return g;
}

// each row of the train_data is an input
void nn_fit(nn_t *nn, const Mat2D *train_data, const Mat2D *labels, size_t batch_size, double lr) {
  assert(nn_output(nn).rows == labels->cols && train_data->rows == labels->rows);
  assert(batch_size != 0);
  nn_t total_g = nn_copy_structure(nn);
  nn_init_zero(&total_g);

  for (size_t i = 0; i < train_data->rows; ++i) {
    Mat2D x = (Mat2D) {
      .cols = 1,
      .rows = train_data->cols,
      .elems = &train_data->elems[i * train_data->cols],
    };

    Mat2D y = (Mat2D) {
      .cols = 1,
      .rows = labels->cols,
      .elems = &labels->elems[i * labels->cols],
    };

    nn_forward(nn, &x);
    nn_t g = nn_backprop(nn, &y);
    nn_add_gradient(&total_g, &g);

    if (i % batch_size == 0) {
      nn_learn(nn, &total_g, batch_size, lr);
      nn_init_zero(&total_g);
    }

    nn_destroy(&g);
  }

  nn_destroy(&total_g);
}
