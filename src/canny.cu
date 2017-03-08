
#include <vector>
#include <math.h>
#include "canny.h"

#define _USE_MATH_DEFINES
#define STRONG_EDGE 0xFFFF
#define NON_EDGE 0x0

// TODO handle overlapping pixels (strong edges, neighbors)
// TODO nice viso c

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

//*****************************************************************************************
// CUDA Hysteresis Implementation
//*****************************************************************************************

///
/// \brief This is a helper function that runs on the GPU.
///
/// It checks if the eight immediate neighbors of a pixel at a given index are above
/// a low threshold, and if they are, sets them to strong edges. This effectively
/// connects the edges.
///
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

    if (in_pixels[nw] >= t_low) {
        out_pixels[nw] = STRONG_EDGE;
    }
    if (in_pixels[n] >= t_low) {
        out_pixels[n] = STRONG_EDGE;
    }
    if (in_pixels[ne] >= t_low) {
        out_pixels[ne] = STRONG_EDGE;
    }
    if (in_pixels[w] >= t_low) {
        out_pixels[w] = STRONG_EDGE;
    }
    if (in_pixels[e] >= t_low) {
        out_pixels[e] = STRONG_EDGE;
    }
    if (in_pixels[sw] >= t_low) {
        out_pixels[sw] = STRONG_EDGE;
    }
    if (in_pixels[s] >= t_low) {
        out_pixels[s] = STRONG_EDGE;
    }
    if (in_pixels[se] >= t_low) {
        out_pixels[se] = STRONG_EDGE;
    }
}

///
/// \brief CUDA implementation of Canny hysteresis high thresholding.
///
/// This kernel is the first pass in the parallel hysteresis step.
/// It launches a thread for every pixel and checks if the value of that pixel
/// is above a high threshold. If it is, the thread marks it as a strong edge (set to 1)
/// in a pixel map and sets the value to the channel max. If it is not, the thread sets
/// the pixel map at the index to 0 and zeros the output buffer space at that index.
///
/// The output of this step is a mask of strong edges and an output buffer with white values
/// at the mask indices which are set.
///
__global__
void cu_hysteresis_high(unsigned *strong_edge_mask, pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, 
                        pixel_channel_t t_high, unsigned img_height, unsigned img_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (img_height * img_width)) {
        /* apply high threshold */
        if (in_pixels[idx] > t_high) {
            strong_edge_mask[idx] = 1;
            out_pixels[idx] = STRONG_EDGE;
        } else {
            strong_edge_mask[idx] = 0;
            out_pixels[idx] = NON_EDGE;
        }
    }
}

///
/// \brief CUDA implementation of Canny hysteresis low thresholding.
///
/// This kernel is the second pass in the parallel hysteresis step. 
/// It launches a thread for every pixel, but skips the first and last rows and columns.
/// For surviving threads, the pixel at the thread ID index is checked to see if it was 
/// previously marked as a strong edge in the first pass. If it was, the thread checks 
/// their eight immediate neighbors and connects them (marks them as strong edges)
/// if the neighbor is above the low threshold.
///
/// The output of this step is an output buffer with both "strong" and "connected" edges
/// set to whtie values. This is the final edge detected image.
///
__global__
void cu_hysteresis_low(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, unsigned *strong_edge_mask,
                        unsigned t_low, unsigned img_height, unsigned img_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx > img_width)                               /* skip first row */
        && (idx < (img_height * img_width) - img_width) /* skip last row */
        && ((idx % img_width) < (img_width - 1))        /* skip last column */
        && ((idx % img_width) > (0)) )                  /* skip first column */
    {
        if (1 == strong_edge_mask[idx]) { /* if this pixel was previously found to be a strong edge */
            trace_immed_neighbors(out_pixels, in_pixels, idx, t_low, img_width);
        }
    }
}

void cu_test_hysteresis(pixel_channel_t *in, pixel_channel_t *out, unsigned rows, unsigned cols)
{
    pixel_channel_t *in_pixels, *out_pixels;
    unsigned *idx_map;

    /* allocate device memory */
    cudaMalloc((void**) &in_pixels, rows*cols*sizeof(pixel_channel_t));
    cudaMalloc((void**) &out_pixels, rows*cols*sizeof(pixel_channel_t));
    cudaMalloc((void**) &idx_map, rows*cols*sizeof(idx_map[0]));

    /* copy original pixels to GPU device as in_pixels*/
    cudaMemcpy(in_pixels, in, rows*cols*sizeof(pixel_channel_t), cudaMemcpyHostToDevice);
      
    pixel_channel_t t_high = 0xFCC;
    pixel_channel_t t_low = 0x1FF;

    /* create task stream to sequence kernels */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* launch kernels */
    cu_hysteresis_high<<<(rows*cols)/1024, 1024, 0, stream>>>(idx_map, out_pixels, in_pixels, t_high, rows, cols);
    cu_hysteresis_low<<<(rows*cols)/1024, 1024, 0, stream>>>(out_pixels, in_pixels, idx_map, t_low, rows, cols);

    /* copy blurred pixels from GPU device back to host as out_pixels*/
    cudaMemcpy(out, out_pixels, rows*cols*sizeof(pixel_channel_t), cudaMemcpyDeviceToHost);

    cudaFree(in_pixels);
    cudaFree(out_pixels);
    cudaFree(idx_map);
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
    //pixel_channel_t t_high = 0xFCC;
    //pixel_channel_t t_low = 0xF5;
    //cu_apply_hysteresis<<<(rows*cols)/1024, 1024>>>(out_pixels, in_pixels, t_high, t_low, rows, cols);

    /* copy blurred pixels from GPU device back to host as out_pixels*/
    cudaMemcpy(orig_pixels, out_pixels, input_pixel_length * sizeof(pixel_t), cudaMemcpyDeviceToHost);

    std::free(blurKernel);
    cudaFree(cudaBlurKernel);
    cudaFree(in_pixels);
    cudaFree(out_pixels);
}

