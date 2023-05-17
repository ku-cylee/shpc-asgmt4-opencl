__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  float sum = 0.0f;
  for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}
