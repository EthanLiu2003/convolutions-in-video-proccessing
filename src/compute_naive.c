#include "compute.h"

// Computes the convolution of two matrices

matrix_t* allocate_output_matrix(int rows, int cols) {
    matrix_t* matrix = (matrix_t*)malloc(sizeof(matrix_t));
    if (!matrix) return NULL;
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (int*)malloc(rows * cols * sizeof(int));
    if (!matrix->data) {
        free(matrix);
        return NULL;
    }
    return matrix;
}

void flip_matrix(matrix_t* b_matrix, matrix_t* flipped_b) {
    for (int i = 0; i < b_matrix->rows; ++i) {
        for (int j = 0; j < b_matrix-> cols; ++j) {
            flipped_b->data[(b_matrix->rows - i - 1) * b_matrix->cols + (b_matrix->cols - j - 1)] = b_matrix->data[i * b_matrix->cols + j];
        }
    }
}

int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  // TODO: convolve matrix a and matrix b, and store the resulting matrix in
  // output_matrix
  matrix_t* flipped_b = allocate_output_matrix(b_matrix->rows, b_matrix->cols);
  if (!flipped_b) return -1;
  flip_matrix(b_matrix, flipped_b);
 
  int output_rows = a_matrix->rows - b_matrix->rows + 1;
  int output_cols = a_matrix->cols - b_matrix->cols + 1;

  *output_matrix = allocate_output_matrix(output_rows, output_cols);
  if (!*output_matrix) {
      free(flipped_b->data);
      free(flipped_b);
      return -1;
  }

  for (int i = 0; i < output_rows; ++i) {
      for (int j = 0; j < output_cols; ++j) {
          int sum = 0;
          for (int k = 0; k < b_matrix->rows; ++k) {
              for (int l = 0; l < b_matrix->cols; ++l) {
                  sum += a_matrix->data[(i + k) * a_matrix->cols + (j + l)] * flipped_b->data[k * b_matrix->cols + l];
              }
          }
          (*output_matrix)->data[i * output_cols + j] = sum;
      }
  }
  free(flipped_b->data);
  free(flipped_b);
  return 0;
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
