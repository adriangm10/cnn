#include "cnn.h"
#include "mat.h"
#include "test_utils.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>

#define NN_OUTPUT(nn) nn.layers[nn.layer_count - 1].dl.a

void forward_test() {
  nn_t nn = new_nn(2, 1, 1);

  nn_add_dense_layer(&nn, 2, RELU);
  nn_add_dense_layer(&nn, 1, RELU);
  nn_compile(&nn);

  assert(nn.layers[1].dl.ws.rows == 2);
  assert(nn.layers[1].dl.ws.cols == 2);
  assert(nn.layers[1].dl.a.rows == 2);
  assert(nn.layers[1].dl.a.cols == 1);

  assert(nn.layers[2].dl.ws.rows == 2);
  assert(nn.layers[2].dl.ws.cols == 1);
  assert(nn.layers[2].dl.a.rows == 1);
  assert(nn.layers[2].dl.a.cols == 1);

  nn.layers[1].dl.bias = -3.0;
  nn.layers[2].dl.bias = 0.0;

  MAT2D_GET(nn.layers[1].dl.ws, 0, 0) = 0.0;
  MAT2D_GET(nn.layers[1].dl.ws, 0, 1) = 1.0;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 0) = 1.0;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 1) = 1.0;

  MAT2D_GET(nn.layers[2].dl.ws, 0, 0) = -1.0;
  MAT2D_GET(nn.layers[2].dl.ws, 0, 1) = 1.0;

  double i1[] = { 0.0, 1.0 };
  double i2[] = { 1.0, 3.0 };

  Mat2D input = (Mat2D) {
    .rows = 2,
    .cols = 1,
    .elems = i1,
  };

  nn_forward(&nn, &input, 1);

  Mat2D out = NN_OUTPUT(nn);
  assert(out.elems[0] == 0.0);

  input.elems = i2;
  nn_forward(&nn, &input, 1);

  out = NN_OUTPUT(nn);
  assert(out.elems[0] == 1.0);

  nn_destroy(&nn);
}

void backprop_test() {
  nn_t nn = new_nn(2, 1, 1);
  nn_add_dense_layer(&nn, 2, SIGMOID);
  nn_add_dense_layer(&nn, 2, SIGMOID);
  nn_compile(&nn);

  assert(nn.layer_count == 3); //input, dense, dense
  assert(nn.layers[1].dl.ws.rows == 2);
  assert(nn.layers[1].dl.ws.cols == 2);
  assert(nn.layers[1].dl.a.rows == 2);
  assert(nn.layers[1].dl.a.cols == 1);

  assert(nn.layers[2].dl.ws.rows == 2);
  assert(nn.layers[2].dl.ws.cols == 2);
  assert(nn.layers[2].dl.a.rows == 2);
  assert(nn.layers[2].dl.a.cols == 1);

  nn.layers[1].dl.bias = .35;
  nn.layers[2].dl.bias = .6;

  MAT2D_GET(nn.layers[1].dl.ws, 0, 0) = 0.15;
  MAT2D_GET(nn.layers[1].dl.ws, 0, 1) = 0.25;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 0) = 0.2;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 1) = 0.3;

  MAT2D_GET(nn.layers[2].dl.ws, 0, 0) = 0.4;
  MAT2D_GET(nn.layers[2].dl.ws, 0, 1) = 0.5;
  MAT2D_GET(nn.layers[2].dl.ws, 1, 0) = 0.45;
  MAT2D_GET(nn.layers[2].dl.ws, 1, 1) = 0.55;

  double i1[] = { .05, .1 };

  Mat2D input = (Mat2D) {
    .cols = 1,
    .rows = 2,
    .elems = i1,
  };

  nn_forward(&nn, &input, 1);
  assert(nn.layers[0].il.input != NULL);
  assert(fabs(nn.layers[1].dl.a.elems[0] - 0.593269992) <= 5e-9 && fabs(nn.layers[1].dl.a.elems[1] - 0.596884378) <= 5e-9);
  Mat2D out = NN_OUTPUT(nn);
  assert(fabs(out.elems[0] - 0.75136507) <= 5e-9 && fabs(out.elems[1] - 0.772928465) <= 5e-9);

  double y1[] = { 0.01, 0.99 };
  Mat2D y = (Mat2D) {
    .cols = 1,
    .rows = 2,
    .elems = y1,
  };

  nn_t g = nn_backprop(&nn, &y);

  printf("2g_11: %.12f\n", g.layers[2].dl.ws.elems[0]);
  printf("2g_12: %.12f\n", g.layers[2].dl.ws.elems[1]);
  printf("2g_21: %.12f\n", g.layers[2].dl.ws.elems[2]);
  printf("2g_22: %.12f\n\n", g.layers[2].dl.ws.elems[3]);

  assert(fabs(g.layers[2].dl.ws.elems[0] - (0.082167041)) <= 5e-9);
  assert(fabs(g.layers[2].dl.ws.elems[1] - (-0.022602540)) <= 5e-9);
  assert(fabs(g.layers[2].dl.ws.elems[2] - (0.082667628)) <= 5e-9);
  assert(fabs(g.layers[2].dl.ws.elems[3] - (-0.022740242)) <= 5e-9);

  printf("1g_11: %.12f\n", g.layers[1].dl.ws.elems[0]);
  printf("1g_12: %.12f\n", g.layers[1].dl.ws.elems[1]);
  printf("1g_21: %.12f\n", g.layers[1].dl.ws.elems[2]);
  printf("1g_22: %.12f\n\n", g.layers[1].dl.ws.elems[3]);

  assert(fabs(g.layers[1].dl.ws.elems[0] - 0.000438568) <= 6e-9);
  assert(fabs(g.layers[1].dl.ws.elems[1] - 0.000497712) <= 6e-9);
  assert(fabs(g.layers[1].dl.ws.elems[2] - 0.000877139) <= 6e-9);
  assert(fabs(g.layers[1].dl.ws.elems[3] - 0.000995420) <= 6e-9);

  nn_destroy(&nn);
  nn_destroy(&g);
}

void fit_test() {
  nn_t nn = new_nn(2, 1, 1);
  nn_add_dense_layer(&nn, 2, SIGMOID);
  nn_add_dense_layer(&nn, 2, SIGMOID);
  nn_compile(&nn);

  nn.layers[1].dl.bias = .35;
  nn.layers[2].dl.bias = .6;

  MAT2D_GET(nn.layers[1].dl.ws, 0, 0) = 0.15;
  MAT2D_GET(nn.layers[1].dl.ws, 0, 1) = 0.25;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 0) = 0.2;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 1) = 0.3;

  MAT2D_GET(nn.layers[2].dl.ws, 0, 0) = 0.4;
  MAT2D_GET(nn.layers[2].dl.ws, 0, 1) = 0.5;
  MAT2D_GET(nn.layers[2].dl.ws, 1, 0) = 0.45;
  MAT2D_GET(nn.layers[2].dl.ws, 1, 1) = 0.55;

  double i1[] = { .05, .1 };
  Mat2D input = (Mat2D) {
    .rows = 1,
    .cols = 2,
    .elems = i1,
  };

  double labels[] = { .01, .99 };
  Mat2D y = (Mat2D) {
    .rows = 1,
    .cols = 2,
    .elems = labels,
  };

  nn_fit(&nn, &input, &y, 1, 0.5);

  assert(fabs(nn.layers[2].dl.ws.elems[0] - .358916480) <= 5e-9);
  assert(fabs(nn.layers[2].dl.ws.elems[1] - .511301270) <= 5e-9);
  assert(fabs(nn.layers[2].dl.ws.elems[2] - .408666186) <= 5e-9);
  assert(fabs(nn.layers[2].dl.ws.elems[3] - .561370121) <= 5e-9);

  assert(fabs(nn.layers[1].dl.ws.elems[0] - .149780716) <= 5e-9);
  assert(fabs(nn.layers[1].dl.ws.elems[1] - .24975114) <= 5e-9);
  assert(fabs(nn.layers[1].dl.ws.elems[2] - .19956143) <= 5e-9);
  assert(fabs(nn.layers[1].dl.ws.elems[3] - .29950229) <= 5e-9);

  nn_destroy(&nn);
}

void compile_test() {
  const int HEIGHT = 28;
  const int WIDTH = 28;
  const int CHANNELS = 3;

  nn_t nn = new_nn(HEIGHT, WIDTH, CHANNELS);

  nn_add_conv2d_layer(&nn, 5, 3, 3, 0, 1, RELU);
  nn_add_avg_pooling_layer(&nn, 2);
  nn_add_conv2d_layer(&nn, 3, 3, 5, 0, 1, RELU);
  nn_add_max_pooling_layer(&nn, 2);
  nn_add_flatten_layer(&nn);

  nn_compile(&nn);

  assert(nn.layers[1].cl.a[0].cols == 26 && nn.layers[1].cl.a[0].rows == 26);

  assert(nn.layers[2].pl.channels == 5);
  assert(nn.layers[2].pl.a[0].cols == 13 && nn.layers[2].pl.a[0].rows == 13);

  assert(nn.layers[3].cl.a[0].cols == 11 && nn.layers[3].cl.a[0].rows == 11);

  assert(nn.layers[4].pl.channels == 3);
  assert(nn.layers[4].pl.a[0].cols == 5 && nn.layers[4].pl.a[0].rows == 5);

  assert(nn.layers[5].fl.a.cols == 1 && nn.layers[5].fl.a.rows == 75);

  nn_destroy(&nn);
}

void conv_forward_test() {
  const int HEIGHT = 28;
  const int WIDTH = 28;
  const int CHANNELS = 1;
  double img[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 148, 210, 253, 253, 113, 87, 148, 55, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 232, 252, 253, 189, 210, 252, 252, 253, 168, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 57, 242, 252, 190, 65, 5, 12, 182, 252, 253, 116, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 252, 252, 183, 14, 0, 0, 92, 252, 252, 225, 21, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 132, 253, 252, 146, 14, 0, 0, 0, 215, 252, 252, 79, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 126, 253, 247, 176, 9, 0, 0, 8, 78, 245, 253, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 16, 232, 252, 176, 0, 0, 0, 36, 201, 252, 252, 169, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 22, 252, 252, 30, 22, 119, 197, 241, 253, 252, 251, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 16, 231, 252, 253, 252, 252, 252, 226, 227, 252, 231, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 55, 235, 253, 217, 138, 42, 24, 192, 252, 143, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 255, 253, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 253, 252, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 252, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 253, 252, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106, 253, 252, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 255, 253, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 218, 252, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 252, 189, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 184, 252, 170, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 147, 252, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  Mat2D input = {
    .cols = WIDTH,
    .rows = HEIGHT,
    .elems = img,
  };

  nn_t nn = new_nn(HEIGHT, WIDTH, CHANNELS);

  nn_add_conv2d_layer(&nn, 5, 3, CHANNELS, 0, 1, RELU);
  nn_add_avg_pooling_layer(&nn, 2);
  nn_add_conv2d_layer(&nn, 3, 3, 5, 0, 1, RELU);
  nn_add_max_pooling_layer(&nn, 2);
  nn_add_flatten_layer(&nn);

  nn_compile(&nn);
  nn_init_zero(&nn);

  nn_forward(&nn, &input, CHANNELS);

  nn_destroy(&nn);
}

int main(void) {
  test_t tests[] = {forward_test, backprop_test, fit_test, compile_test};
  run_tests(tests, 4);
  return 0;
}
