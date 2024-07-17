#include "cnn.h"
#include "mat.h"
#include <assert.h>
#include <omp.h>
#include <time.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TRAIN_IMGS "./data/train-images-idx3-ubyte"
#define TRAIN_LBLS "./data/train-labels-idx1-ubyte"
#define TEST_IMGS "./data/t10k-images-idx3-ubyte"
#define TEST_LBLS "./data/t10k-labels-idx1-ubyte"
#define IMG_SIZE 784 // 28*28
#define IMG_SIDE 28
#define MAX_TRAIN 60000
#define MAX_TEST 10000
#define LBL_MN 2049
#define IMG_MN 2051
#define MAX_EPOCH 10000

int reverse_int(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 0xFF;
  c2 = (i >> 8) & 0xFF;
  c3 = (i >> 16) & 0xFF;
  c4 = (i >> 24) & 0xFF;

  return (int)((c1 << 24) | (c2 << 16) | (c3 << 8) | c4);
}

// read the labels from file_name
// both ends inclusive,
// starts counting from 1
// if to = -1, take from @from to the end
// each row is of shape [0, 1, ... 0]
Mat2D read_labels(char *file_name, int from, int to) {
  assert((from <= to || to == -1) && from >= 1);
  int magic_number, fd, n;

  fd = open(file_name, O_RDONLY);
  assert(fd != -1 && "could not open the file");

  read(fd, &magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);
  assert(magic_number == LBL_MN && "not a mnist label file");

  read(fd, &n, sizeof(n));
  n = reverse_int(n);
  assert(from < n);

  if (to == -1)
    to = n;
  to = to >= n ? n : to;

  Mat2D m = new_Mat2D(to - from + 1, 10);
  zero_init_Mat2D(&m);

  lseek(fd, (from - 1) * sizeof(unsigned char), SEEK_CUR);

  for (int i = 0; i < to - from + 1; i++) {
    unsigned char temp;
    read(fd, &temp, sizeof(unsigned char));
    MAT2D_GET(m, i, temp) = 1.0;
  }

  close(fd);
  return m;
}

// read the images from file_name
// both ends inclusive,
// starts counting from 1,
// if to = -1, take from @from to the end
// each row is an image
Mat2D read_imgs(char *file_name, int from, int to) {
  assert((from <= to || to == -1) && from >= 1);
  int magic_number, fd, n, rows, cols;

  fd = open(file_name, O_RDONLY);
  assert(fd != -1 && "could not open the file");

  read(fd, &magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);
  assert(magic_number == IMG_MN && "not a mnist image file");

  read(fd, &n, sizeof(int));
  n = reverse_int(n);
  assert(from < n);

  read(fd, &rows, sizeof(int));
  rows = reverse_int(rows);

  read(fd, &cols, sizeof(int));
  cols = reverse_int(cols);
  assert(rows == cols && rows == IMG_SIDE);

  if (to == -1)
    to = n;

  to = to >= n ? n : to;

  Mat2D m = new_Mat2D(to - from + 1, IMG_SIZE);
  lseek(fd, (from - 1) * IMG_SIZE * sizeof(unsigned char), SEEK_CUR);

  for (int i = 0; i < to - from + 1; ++i) {
    for (int j = 0; j < rows; ++j) {
      for (int k = 0; k < cols; ++k) {
        unsigned char temp;
        read(fd, &temp, sizeof(temp));
        MAT2D_GET(m, i, j * IMG_SIDE + k) = (double) temp;
      }
    }
  }

  close(fd);
  return m;
}

void print_mnist(const Mat2D *number, const char *end) {
  char *chars[] = { " ", "░", "▒", "▓", "█" };

  for (size_t i = 0; i < number->rows; ++i) {
    for (size_t j = 0; j < number->cols; ++j) {
      int c = round(MAT2D_GET((*number), i, j) / 255.0 * 4.0);
      printf("%s", chars[c]);
    }
    puts("");
  }
  printf("%s", end);
}

int main(void) {
  omp_set_num_threads(1);
  srandom(time(NULL));
  nn_t mnist_nn = new_nn();
  nn_add_dense_layer(&mnist_nn, IMG_SIZE, 32, SIGMOID);
  nn_add_dense_layer(&mnist_nn, 32, 14, SIGMOID);
  nn_add_dense_layer(&mnist_nn, 14, 10, SOFTMAX);
  nn_init_random(&mnist_nn, -1.0, 1.0);

  Mat2D imgs = read_imgs(TRAIN_IMGS, 1, 25);
  Mat2D labels = read_labels(TRAIN_LBLS, 1, 25);

  int first_img = (float) random() / (float) RAND_MAX * imgs.rows;
  printf("first image: %d\n", first_img);
  double *img = &imgs.elems[first_img * imgs.cols];
  print_mnist(&((Mat2D) {IMG_SIDE, IMG_SIDE, img}), "\n\n");
  printf("first img expected output: ");
  print_Mat2D(&((Mat2D) {labels.cols, 1, &labels.elems[first_img * labels.cols]}), "");

  nn_forward(&mnist_nn, &((Mat2D) {imgs.cols, 1, img}));
  Mat2D o1 = nn_output(&mnist_nn);
  printf("got (before training): ");
  print_Mat2D(&o1, "\n");

  for (int e = 0; e < MAX_EPOCH; ++e) {
    nn_fit(&mnist_nn, &imgs, &labels, 1, 1.0);
  }

  nn_forward(&mnist_nn, &((Mat2D) {imgs.cols, 1, img}));
  o1 = nn_output(&mnist_nn);
  printf("got (after training): ");
  print_Mat2D(&o1, "\n");

  destroy_Mat2D(&labels);
  destroy_Mat2D(&imgs);
  nn_destroy(&mnist_nn);
  return 0;
}
