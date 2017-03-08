
#include "canny.h"
#define _USE_MATH_DEFINES
#include <math.h>
#define EDGE 0xFFFF

__global__
void cu_apply_gaussian_filter(pixel_t *in_pixels, pixel_t *out_pixels, int rows, int cols, double *in_kernel) 
{
    //copy kernel array from global memory to a shared array
    __shared__ double kernel[KERNEL_SIZE][KERNEL_SIZE];
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            kernel[i][j] = in_kernel[i*KERNEL_SIZE + j];
        }
    }
    
    __syncthreads();

    //determine id of thread which corresponds to an individual pixel
    int pixNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixNum >= 0 && pixNum < rows * cols) {
   
        double kernelSum;
        double redPixelVal;
        double greenPixelVal;
        double bluePixelVal;

        //Apply Kernel to each pixel of image
        for (int i = 0; i < KERNEL_SIZE; ++i) {
            for (int j = 0; j < KERNEL_SIZE; ++j) {    
            
                //check edge cases, if within bounds, apply filter
                if (((pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)) >= 0)
                    && ((pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)) <= rows*cols-1)
                    && (((pixNum % cols) + j - ((KERNEL_SIZE-1)/2)) >= 0)
                    && (((pixNum % cols) + j - ((KERNEL_SIZE-1)/2)) <= (cols-1))) {

                    redPixelVal += kernel[i][j] * in_pixels[pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)].red;
                    greenPixelVal += kernel[i][j] * in_pixels[pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)].green;
                    bluePixelVal += kernel[i][j] * in_pixels[pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)].blue;
                    kernelSum += kernel[i][j];
                }
            }
        }
        
        //update output image
        out_pixels[pixNum].red = redPixelVal / kernelSum;
        out_pixels[pixNum].green = greenPixelVal / kernelSum;
        out_pixels[pixNum].blue = bluePixelVal / kernelSum;
    }
}

__device__
void trace_immed_neighbors(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, 
                            unsigned idx, pixel_channel_t t_low, unsigned img_width)
{
    /* directions representing indices of neighbors */
    unsigned n, s, e, w;
    unsigned nw, ne, sw, se;

    /* get indices */
    n = idx - img_width;
    nw = n - 1;
    ne = n + 1;
    s = idx + img_width;
    sw = s - 1;
    se = s + 1;
    w = idx - 1;
    e = idx + 1;

    if ((in_pixels[nw] >= t_low) && (out_pixels[nw] != EDGE)) {
        out_pixels[nw] = EDGE;
    }
    if ((in_pixels[n] >= t_low) && (out_pixels[n] != EDGE)) {
        out_pixels[n] = EDGE;
    }
    if ((in_pixels[ne] >= t_low) && (out_pixels[ne] != EDGE)) {
        out_pixels[ne] = EDGE;
    }
    if ((in_pixels[w] >= t_low) && (out_pixels[w] != EDGE)) {
        out_pixels[w] = EDGE;
    }
    if ((in_pixels[e] >= t_low) && (out_pixels[e] != EDGE)) {
        out_pixels[e] = EDGE;
    }
    if ((in_pixels[sw] >= t_low) && (out_pixels[sw] != EDGE)) {
        out_pixels[sw] = EDGE;
    }
    if ((in_pixels[s] >= t_low) && (out_pixels[s] != EDGE)) {
        out_pixels[s] = EDGE;
    }
    if ((in_pixels[se] >= t_low) && (out_pixels[se] != EDGE)) {
        out_pixels[se] = EDGE;
    }
}

__global__
void cu_apply_hysteresis(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, 
                        pixel_channel_t t_high, pixel_channel_t t_low, 
                        unsigned img_height, unsigned img_width)
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ix; // TODO fix indexing
    if (iy <= img_height && ix <= img_width) {

        // apply high threshold
        if ((in_pixels[idx] > t_high) && (out_pixels[idx] != EDGE)) {
            out_pixels[idx] = EDGE;
        }
        // apply low threshold to neighbors
        trace_immed_neighbors(out_pixels, in_pixels, idx, t_low, img_width);
    }
}

void cu_detect_edges(pixel_t *orig_pixels, int rows, int cols, double kernel[KERNEL_SIZE][KERNEL_SIZE]) 
{
    pixel_t *in_pixels, *out_pixels;
    double *blurKernel, *cudaBlurKernel;
    int input_pixel_length = rows * cols;

    blurKernel = (double*) std::malloc(KERNEL_SIZE*KERNEL_SIZE*sizeof(double));

    for (int i = 0; i < KERNEL_SIZE; ++i) {
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            blurKernel[i*KERNEL_SIZE + j] = kernel[i][j];
        }
    }

    /* allocate device memory */
    cudaMalloc((void**) &in_pixels, input_pixel_length*sizeof(pixel_t));
    cudaMalloc((void**) &out_pixels, input_pixel_length*sizeof(pixel_t));
    cudaMalloc((void**) &cudaBlurKernel, KERNEL_SIZE*KERNEL_SIZE*sizeof(double));

    /* copy original pixels to GPU device as in_pixels*/
    cudaMemcpy(in_pixels, orig_pixels, input_pixel_length*sizeof(pixel_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaBlurKernel, blurKernel, KERNEL_SIZE*KERNEL_SIZE*sizeof(double), cudaMemcpyHostToDevice);

      
    cu_apply_gaussian_filter<<<(rows*cols)/1024, 1024>>>(in_pixels, out_pixels, rows, cols, cudaBlurKernel);
    //cu_compute_intensity_gradient();
    //cu_suppress_non_max();
    //cu_apply_double_threshold();
    //cu_apply_hysteresis(pixel_t *out_pixels, pixel_t *in_pixels, pixel_t hi_thld, pixel_t lo_thld);

    /* copy blurred pixels from GPU device back to host as out_pixels*/
    cudaMemcpy(orig_pixels, out_pixels, input_pixel_length * sizeof(pixel_t), cudaMemcpyDeviceToHost);

    std::free(blurKernel);
    cudaFree(cudaBlurKernel);
    cudaFree(in_pixels);
    cudaFree(out_pixels);
}

