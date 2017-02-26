#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = a*x[idx] + y[idx];
}

int main(void) {
    int n = 1 << 20;
    float *x, *y, *d_x, *d_y;

    /* allocate host memory */
    x = (float*)malloc(n*sizeof(float));
    y = (float*)malloc(n*sizeof(float));
    cudaMalloc(&d_x, n*sizeof(float));
    cudaMalloc(&d_y, n*sizeof(float));

    /* init x,y values */
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    /* copy x, y to GPU device */
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

    /* run saxpy GPU kernel */
    saxpy<<<(n+255)/256, 256>>>(n, 2.0f, d_x, d_y);

    /* copy results back to host */
    cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    /* print error margin */
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        max_err = max(max_err, abs(y[i] - 4.0f));
    }
    printf("Max error: %f\n", max_err);

    /* free memory */
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
