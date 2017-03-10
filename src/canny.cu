
#include "canny.h"
#define _USE_MATH_DEFINES
#include <math.h>
#define EDGE 0xFFFF

#define RGB2GRAY_CONST_ARR_SIZE 3

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

///
/// \brief Compute gradient (first order derivative x and y)
///
__global__
void cu_compute_intensity_gradient(pixel_t *in_pixels, pixel_channel_t_signed *deltaX_channel, pixel_channel_t_signed *deltaY_channel, unsigned parser_length, unsigned offset, float *RGB_kernel)
{
    //copy kernel array from global memory to a shared array
    __shared__ float rgb_kernel_shared[RGB2GRAY_CONST_ARR_SIZE];
    for (int i = 0; i < RGB2GRAY_CONST_ARR_SIZE; ++i) {
            rgb_kernel_shared[i] = RGB_kernel[i];
    }
    // compute delta X ***************************
    // deltaX = f(x+1) - f(x-1)
    
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < parser_length * offset)
    {
        int16_t deltaXred;
        int16_t deltaYred;
        int16_t deltaXgreen;
        int16_t deltaYgreen;
        int16_t deltaXblue;
        int16_t deltaYblue;

        if((idx % offset) == 0)
        {
            // gradient at the first pixel of each line
            // note: at the edge pix[idx-1] does NOT exist
            deltaXred = (int16_t)(in_pixels[idx+1].red - in_pixels[idx].red);
            deltaXgreen = (int16_t)(in_pixels[idx+1].green - in_pixels[idx].green);
            deltaXblue = (int16_t)(in_pixels[idx+1].blue - in_pixels[idx].blue);
            // gradient at the first pixel of each line
            // note: at the edge pix[idx-1] does NOT exist
            deltaYred = (int16_t)(in_pixels[idx+offset].red - in_pixels[idx].red);
            deltaYgreen = (int16_t)(in_pixels[idx+offset].green - in_pixels[idx].green);
            deltaYblue = (int16_t)(in_pixels[idx+offset].blue - in_pixels[idx].blue);
        }
        else if((idx % offset) == (offset - 1))
        {
            deltaXred = (int16_t)(in_pixels[idx].red - in_pixels[idx-1].red);
            deltaXgreen = (int16_t)(in_pixels[idx].green - in_pixels[idx-1].green);
            deltaXblue = (int16_t)(in_pixels[idx].blue - in_pixels[idx-1].blue);
            deltaYred = (int16_t)(in_pixels[idx].red - in_pixels[idx-offset].red);
            deltaYgreen = (int16_t)(in_pixels[idx].green - in_pixels[idx-offset].green);
            deltaYblue = (int16_t)(in_pixels[idx].blue - in_pixels[idx-offset].blue);
        }
        // gradients where NOT edge
        else
        {
            deltaXred = (int16_t)(in_pixels[idx+1].red - in_pixels[idx-1].red);
            deltaXgreen = (int16_t)(in_pixels[idx+1].green - in_pixels[idx-1].green);
            deltaXblue = (int16_t)(in_pixels[idx+1].blue - in_pixels[idx-1].blue);
            deltaYred = (int16_t)(in_pixels[idx+offset].red - in_pixels[idx-offset].red);
            deltaYgreen = (int16_t)(in_pixels[idx+offset].green - in_pixels[idx-offset].green);
            deltaYblue = (int16_t)(in_pixels[idx+offset].blue - in_pixels[idx-offset].blue);
        }
	//deltaY_channel[idx] = deltaYred;
        //deltaX_channel[idx] = deltaXred;
        deltaX_channel[idx] = (int16_t)(rgb_kernel_shared[0] * deltaXred + rgb_kernel_shared[1] * deltaXgreen + rgb_kernel_shared[2] * deltaXblue);
        deltaY_channel[idx] = (int16_t)(rgb_kernel_shared[0] * deltaYred + rgb_kernel_shared[1] * deltaYgreen + rgb_kernel_shared[2] * deltaYblue); 
    }
}


///
/// \brief Compute magnitude of gradient(deltaX & deltaY) per pixel.
///
__global__
void cu_magnitude(pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *out_pixel, unsigned parser_length, unsigned offset)
{
    //computation
    //Assigned a thread to each pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < parser_length * offset) {
            out_pixel[idx] =  (pixel_channel_t)(sqrt((double)deltaX[idx]*deltaX[idx] + 
                            (double)deltaY[idx]*deltaY[idx]) + 0.5);
        }
}

///
/// \brief Non Maximal Suppression
/// If the centre pixel is not greater than neighboured pixels in the direction,
/// then the center pixel is set to zero.
/// This process results in one pixel wide ridges.
///
__global__
void cu_suppress_non_max(pixel_channel_t *mag, pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *nms, unsigned parser_length, unsigned offset)
{
   
    const pixel_channel_t SUPPRESSED = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < parser_length * offset)
    {
        float alpha;
        float mag1, mag2;
        // put zero all boundaries of image
        // TOP edge line of the image
        if((idx >= 0) && (idx <offset))
            nms[idx] = 0;

        // BOTTOM edge line of image
        else if((idx >= (parser_length-1)*offset) && (idx < (offset * parser_length)))
            nms[idx] = 0;

        // LEFT & RIGHT edge line
        else if(((idx % offset)==0) || ((idx % offset)==(offset - 1)))
        {
            nms[idx] = 0;
        }

        else // not the boundaries
        {
            // if magnitude = 0, no edge
            if(mag[idx] == 0)
                nms[idx] = SUPPRESSED;
                else{
                    if(deltaX[idx] >= 0)
                    {
                        if(deltaY[idx] >= 0)  // dx >= 0, dy >= 0
                        {
                            if((deltaX[idx] - deltaY[idx]) >= 0)       // direction 1 (SEE, South-East-East)
                            {
                                alpha = (float)deltaY[idx] / deltaX[idx];
                                mag1 = (1-alpha)*mag[idx+1] + alpha*mag[idx+offset+1];
                                mag2 = (1-alpha)*mag[idx-1] + alpha*mag[idx-offset-1];
                            }
                            else                                // direction 2 (SSE)
                            {
                                alpha = (float)deltaX[idx] / deltaY[idx];
                                mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset+1];
                                mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset-1];
                            }
                        }
                        else  // dx >= 0, dy < 0
                        {
                            if((deltaX[idx] + deltaY[idx]) >= 0)    // direction 8 (NEE)
                            {
                                alpha = (float)-deltaY[idx] / deltaX[idx];
                                mag1 = (1-alpha)*mag[idx+1] + alpha*mag[idx-offset+1];
                                mag2 = (1-alpha)*mag[idx-1] + alpha*mag[idx+offset-1];
                            }
                            else                                // direction 7 (NNE)
                            {
                                alpha = (float)deltaX[idx] / -deltaY[idx];
                                mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset-1];
                                mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset+1];
                            }
                        }
                    }

                    else
                    {
                        if(deltaY[idx] >= 0) // dx < 0, dy >= 0
                        {
                            if((deltaX[idx] + deltaY[idx]) >= 0)    // direction 3 (SSW)
                            {
                                alpha = (float)-deltaX[idx] / deltaY[idx];
                                mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset-1];
                                mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset+1];
                            }
                            else                                // direction 4 (SWW)
                            {
                                alpha = (float)deltaY[idx] / -deltaX[idx];
                                mag1 = (1-alpha)*mag[idx-1] + alpha*mag[idx+offset-1];
                                mag2 = (1-alpha)*mag[idx+1] + alpha*mag[idx-offset+1];
                            }
                        }

                        else // dx < 0, dy < 0
                        {
                             if((-deltaX[idx] + deltaY[idx]) >= 0)   // direction 5 (NWW)
                             {
                                 alpha = (float)deltaY[idx] / deltaX[idx];
                                 mag1 = (1-alpha)*mag[idx-1] + alpha*mag[idx-offset-1];
                                 mag2 = (1-alpha)*mag[idx+1] + alpha*mag[idx+offset+1];
                             }
                             else                                // direction 6 (NNW)
                             {
                                 alpha = (float)deltaX[idx] / deltaY[idx];
                                 mag1 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset-1];
                                 mag2 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset+1];
                             }
                        }
                    }

                    // non-maximal suppression
                    // compare mag1, mag2 and mag[t]
                    // if mag[t] is smaller than one of the neighbours then suppress it
                    if((mag[idx] < mag1) || (mag[idx] < mag2))
                         nms[idx] = SUPPRESSED;
                    else
                    {
                         nms[idx] = mag[idx];
                    }

            } // END OF ELSE (mag != 0)
        } // END OF FOR(j)
    } // END OF FOR(i)
}


void cu_test_nonmax(pixel_channel_t *mag, pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *nms, unsigned rows, unsigned cols)
{
    pixel_channel_t *magnitude_v;
    pixel_channel_t *d_nms;
    pixel_channel_t_signed *deltaX_gray;
    pixel_channel_t_signed *deltaY_gray;

    cudaMalloc((void**) &magnitude_v, sizeof(pixel_channel_t)*rows*cols); 
    cudaMalloc((void**) &d_nms, sizeof(pixel_channel_t)*rows*cols); 
    cudaMalloc((void**) &deltaX_gray, sizeof(pixel_channel_t_signed)*rows*cols);
    cudaMalloc((void**) &deltaY_gray, sizeof(pixel_channel_t_signed)*rows*cols);

    cudaMemcpy(magnitude_v, mag, rows*cols*sizeof(pixel_channel_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deltaX_gray, deltaX, rows*cols*sizeof(pixel_channel_t_signed), cudaMemcpyHostToDevice);
    cudaMemcpy(deltaY_gray, deltaY, rows*cols*sizeof(pixel_channel_t_signed), cudaMemcpyHostToDevice);

    cu_suppress_non_max<<<(rows*cols)/1024, 1024>>>(magnitude_v, deltaX_gray, deltaY_gray, d_nms, rows, cols);

    cudaMemcpy(nms, d_nms, rows*cols*sizeof(pixel_channel_t), cudaMemcpyDeviceToHost);
    
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

void cu_test_mag(pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *out_pixel, unsigned rows, unsigned cols)
{
    pixel_channel_t *magnitude_v;
    pixel_channel_t_signed *deltaX_gray;
    pixel_channel_t_signed *deltaY_gray;

    cudaMalloc((void**) &magnitude_v, sizeof(pixel_channel_t)*rows*cols); 
    cudaMalloc((void**) &deltaX_gray, sizeof(pixel_channel_t_signed)*rows*cols);
    cudaMalloc((void**) &deltaY_gray, sizeof(pixel_channel_t_signed)*rows*cols);

    cudaMemcpy(deltaX_gray, deltaX, rows*cols*sizeof(pixel_channel_t_signed), cudaMemcpyHostToDevice);
    cudaMemcpy(deltaY_gray, deltaY, rows*cols*sizeof(pixel_channel_t_signed), cudaMemcpyHostToDevice);
    cu_magnitude<<<(rows*cols)/1024, 1024>>>(deltaX_gray, deltaY_gray, magnitude_v, rows, cols);
    cudaMemcpy(out_pixel, magnitude_v, rows*cols*sizeof(pixel_channel_t_signed), cudaMemcpyDeviceToHost);

}

void cu_detect_edges(pixel_t *orig_pixels, int rows, int cols, double kernel[KERNEL_SIZE][KERNEL_SIZE]) 
{
    pixel_t *in_pixels, *out_pixels;
    //double *blurKernel, *cudaBlurKernel;
    int input_pixel_length = rows * cols;
    float *RGB2GrayKernel;
    float RGB2Gray[RGB2GRAY_CONST_ARR_SIZE] = {0.2989, 0.5870, 0.1140};
    
    pixel_channel_t *magnitude_v;
    pixel_channel_t_signed *deltaX_gray;
    pixel_channel_t_signed *deltaY_gray;
    pixel_channel_t *magnitude_v_h;
    pixel_channel_t_signed *deltaX_gray_h;
    pixel_channel_t_signed *deltaY_gray_h;
    pixel_t *out_pixels_test;
    pixel_channel_t *out_channel_test;

    //pixel_t *out_pixels_test = new pixel_t[input_pixel_length];
    //pixel_channel_t *out_channel_test = new pixel_channel_t[input_pixel_length];

    //blurKernel = (double*) std::malloc(KERNEL_SIZE*KERNEL_SIZE*sizeof(double));

    /*for (int i = 0; i < KERNEL_SIZE; ++i) {
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            blurKernel[i*KERNEL_SIZE + j] = kernel[i][j];
        }
    }*/
    out_pixels_test = (pixel_t*) std::malloc(input_pixel_length*sizeof(pixel_t));
    out_channel_test = (pixel_channel_t*) std::malloc(input_pixel_length*sizeof(pixel_channel_t));
    magnitude_v_h = (pixel_channel_t*) std::malloc(input_pixel_length*sizeof(pixel_channel_t));
    deltaX_gray_h = (pixel_channel_t_signed*) std::malloc(input_pixel_length*sizeof(pixel_channel_t_signed));
    deltaY_gray_h = (pixel_channel_t_signed*) std::malloc(input_pixel_length*sizeof(pixel_channel_t_signed));

    /* allocate device memory */
    cudaMalloc((void**) &in_pixels, input_pixel_length*sizeof(pixel_t));
    cudaMalloc((void**) &out_pixels, input_pixel_length*sizeof(pixel_t));
    cudaMalloc((void**) &magnitude_v, sizeof(pixel_channel_t)*input_pixel_length); 
    cudaMalloc((void**) &deltaX_gray, sizeof(pixel_channel_t_signed)*input_pixel_length);
    cudaMalloc((void**) &deltaY_gray, sizeof(pixel_channel_t_signed)*input_pixel_length);
    cudaMalloc((void**) &RGB2GrayKernel, RGB2GRAY_CONST_ARR_SIZE*sizeof(float));
    
    /* copy original pixels to GPU device as in_pixels*/
    cudaMemcpy(in_pixels, orig_pixels, input_pixel_length*sizeof(pixel_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(cudaBlurKernel, blurKernel, KERNEL_SIZE*KERNEL_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(RGB2GrayKernel, RGB2Gray, RGB2GRAY_CONST_ARR_SIZE*sizeof(float), cudaMemcpyHostToDevice);
      
    //cu_apply_gaussian_filter<<<(rows*cols)/1024, 1024>>>(in_pixels, out_pixels, rows, cols, cudaBlurKernel);
    cu_compute_intensity_gradient<<<(rows*cols)/1024, 1024>>>(in_pixels, deltaX_gray, deltaY_gray, rows, cols, RGB2GrayKernel);
    cudaMemcpy(deltaX_gray_h, deltaX_gray, input_pixel_length * sizeof(pixel_channel_t_signed), cudaMemcpyDeviceToHost);
    for(int i = 0; i < input_pixel_length; i++)
        printf("deltaX[%d]: %d\n",i, deltaX_gray_h[i]);
    cudaMemcpy(deltaY_gray_h, deltaY_gray, input_pixel_length * sizeof(pixel_channel_t_signed), cudaMemcpyDeviceToHost);
    cudaMemcpy(deltaX_gray, deltaX_gray_h, input_pixel_length*sizeof(pixel_channel_t_signed), cudaMemcpyHostToDevice);
    cudaMemcpy(deltaY_gray, deltaY_gray_h, input_pixel_length*sizeof(pixel_channel_t_signed), cudaMemcpyHostToDevice);
    cu_magnitude<<<(rows*cols)/1024, 1024>>>(deltaX_gray, deltaY_gray, magnitude_v, rows, cols);

    //cu_apply_double_threshold();
    //cu_apply_hysteresis(pixel_t *out_pixels, pixel_t *in_pixels, pixel_t hi_thld, pixel_t lo_thld);

    /* copy blurred pixels from GPU device back to host as out_pixels*/
    //cudaMemcpy(orig_pixels, out_pixels, input_pixel_length * sizeof(pixel_t), cudaMemcpyDeviceToHost);
    //out_pixels_test
    cudaMemcpy(out_channel_test, deltaX_gray_h, input_pixel_length * sizeof(pixel_channel_t), cudaMemcpyDeviceToHost);
    for(int i = 0; i < input_pixel_length; i++)
    {
        out_pixels_test[i].red = (int16_t)out_channel_test[i];
        out_pixels_test[i].green = (int16_t)out_channel_test[i];
        out_pixels_test[i].blue = (int16_t)out_channel_test[i];
    }
    memcpy(orig_pixels, out_pixels_test, input_pixel_length * sizeof(pixel_t)); 
    //std::free(blurKernel);
    //cudaFree(cudaBlurKernel);
    cudaFree(in_pixels);
    cudaFree(out_pixels);
    cudaFree(magnitude_v);
    cudaFree(deltaX_gray);
    cudaFree(deltaY_gray);
    std::free(out_pixels_test);
    std::free(out_channel_test);
    std::free(magnitude_v_h);
    std::free(deltaX_gray_h);
    std::free(deltaY_gray_h);
    
    //cudaFree(threshold_pixels);
    //cudaFree(out_pixels_test);
    //delete [] out_pixels_test;
    //delete [] out_channel_test;
}

