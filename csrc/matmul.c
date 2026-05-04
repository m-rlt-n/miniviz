/*
 * matmul.c — naive float32 matrix multiplication.
 *
 * This is the baseline implementation we'll measure against later when we
 * introduce cache blocking and SIMD. The triple loop is the textbook (i, j, k)
 * ordering: for each output element C[i,j], we compute the dot product across
 * the shared dimension k.
 *
 * Layout convention: all matrices are row-major float32, matching NumPy's
 * default for ndarray of dtype=float32 with C-contiguous memory. That means
 * element (row, col) of a matrix with N columns lives at offset row*N + col.
 *
 * Shapes:
 *   A: M x K
 *   B: K x N
 *   C: M x N
 *
 * The caller is responsible for allocating C (it will be overwritten, not
 * accumulated into).
 */

#include <stddef.h>

void matmul_naive_f32(
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t K
) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}
