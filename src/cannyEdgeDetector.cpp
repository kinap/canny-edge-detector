
#include <iostream> // cout, cerr
#include <assert.h> // assert
#include <string.h> // memcpy
#define _USE_MATH_DEFINES
#include <math.h>
#include "cannyEdgeDetector.hpp"
#include "canny.h"

CannyEdgeDetector::CannyEdgeDetector(std::shared_ptr<ImgMgr> image)
: EdgeDetector(image)
{
    /* a strong edge is the largest value a channel can hold */
    /* e.g. 8 bit channel: (1 << 8) - 1 -> b1_0000_0000 - b1 -> 0_1111_1111 */
    //unsigned max_val = (1 << image->getChannelDepth()) - 1;
    m_edge = 0xFFFF;
}

CannyEdgeDetector::~CannyEdgeDetector(void)
{

}

///
/// \brief Runs the canny edge detection algorithm on an image represented by
/// the image manager instance this edge detection instance was constructed with.
///
void CannyEdgeDetector::detect_edges(bool serial)
{
    pixel_t *orig_pixels = m_image_mgr->getPixelHandle();
    unsigned input_pixel_length = m_image_mgr->getPixelCount();
    int rows = m_image_mgr->getImgHeight();
    int cols = m_image_mgr->getImgWidth();

    //create convolution kernel for Gaussian Blur
    double kernel[KERNEL_SIZE][KERNEL_SIZE];
    double stDev;
    double scaleVal;
    int i;
    int j;

    //populate Convolution Kernel
    stDev = (double)KERNEL_SIZE/3;

    scaleVal = 1;
    for (i = 0; i < KERNEL_SIZE; ++i) {
        for (j = 0; j < KERNEL_SIZE; ++j) {
            double xComp = pow((i - KERNEL_SIZE/2), 2);
            double yComp = pow((j - KERNEL_SIZE/2), 2);

            double stDevSq = pow(stDev, 2);
            double pi = M_PI;

            //calculate the value at each index of the Kernel
            double kernelVal = exp(-(((xComp) + (yComp)) / (2 * stDevSq)));
            kernelVal = (1 / (sqrt(2 * pi)*stDev)) * kernelVal;

            //populate Kernel
            kernel[i][j] = kernelVal;

            if (i==0 && j==0) 
            {
                scaleVal = kernel[0][0];
            }

            //normalize Kernel
            kernel[i][j] = kernel[i][j] / scaleVal;
        }
    }

    if (true == serial) {
        std::cout << "  executing serially on CPU" << std::endl;
        /* allocate intermeditate buffers */
        pixel_t *buf0 = new pixel_t[input_pixel_length];
        pixel_channel_t *magnitude_v = new pixel_channel_t[input_pixel_length]; 
        pixel_channel_t_signed *deltaX_gray = new pixel_channel_t_signed[input_pixel_length];
        pixel_channel_t_signed *deltaY_gray = new pixel_channel_t_signed[input_pixel_length];
        pixel_channel_t *threshold_pixels = new pixel_channel_t[input_pixel_length];
        pixel_channel_t *final_pixels = new pixel_channel_t[input_pixel_length];
        pixel_t *final_image = new pixel_t[input_pixel_length];

        assert(nullptr != buf0);
        assert(nullptr != magnitude_v);
        assert(nullptr != deltaX_gray);
        assert(nullptr != deltaY_gray);
        assert(nullptr != threshold_pixels);
        assert(nullptr != final_pixels);
        assert(nullptr != final_image);

        /* run canny edge detection core */
        apply_gaussian_filter(buf0, orig_pixels, kernel);

        //compute_intensity_gradient(buf0, deltaX_gray, deltaY_gray, input_pixel_length);
        cu_test_gradient(buf0, deltaX_gray, deltaY_gray, rows, cols);
        
        //cu_test_mag(deltaX_gray, deltaY_gray, magnitude_v, rows, cols);
        magnitude(deltaX_gray, deltaY_gray, magnitude_v, input_pixel_length);

        //cu_test_nonmax(magnitude_v, deltaX_gray, deltaY_gray, threshold_pixels, rows, cols);
        suppress_non_max(magnitude_v, deltaX_gray, deltaY_gray, threshold_pixels);

        //cu_test_hysteresis(threshold_pixels, final_pixels, m_image_mgr->getImgHeight(), m_image_mgr->getImgWidth());
        pixel_channel_t hi = 0xFCC;
        pixel_channel_t lo = 0xF5;
        apply_hysteresis(final_pixels, threshold_pixels, hi, lo);

        /* convert single channel to grayscale final image */
        unsigned idx = 0;
        unsigned offset = m_image_mgr->getImgWidth();
        unsigned parser_length = m_image_mgr->getImgHeight();
     
        for(unsigned i = 0; i < parser_length; ++i) {
            for(unsigned j = 0; j < offset; ++j, ++idx) {
                final_image[idx].red = final_pixels[idx];
                final_image[idx].green = final_pixels[idx];
                final_image[idx].blue = final_pixels[idx];
            }
        }

        /* copy edge detected image back into image mgr class so we can write it out later */
        memcpy(orig_pixels, final_image, input_pixel_length * sizeof(pixel_t));

        delete [] buf0;
        delete [] magnitude_v;
        delete [] deltaX_gray;
        delete [] deltaY_gray;
        delete [] threshold_pixels;
        delete [] final_pixels;
        delete [] final_image;

    } else { // GPGPU
        std::cout << "  executing in parallel on GPU" << std::endl;
        /* Copy pixels to device - results of each stage stored on GPU and passed to next kernel */
        cu_detect_edges(orig_pixels, rows, cols, kernel);
    }
}

///
///This function is used to slightly blur the image to remove noise
///
void CannyEdgeDetector::apply_gaussian_filter(pixel_t *out_pixels, pixel_t *in_pixels, double kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    /*
    double kernel[KERNEL_SIZE][KERNEL_SIZE];
    double stDev;
    double scaleVal;
    int i;
    int j;

    //populate Convolution Kernel
    stDev = (double)KERNEL_SIZE/3;

        std::cout <<std::dec<< stDev<< std::endl;

    scaleVal = 1;
    for (i = 0; i < KERNEL_SIZE; ++i) {
        for (j = 0; j < KERNEL_SIZE; ++j) {
            double xComp = pow((i - KERNEL_SIZE/2), 2);
            double yComp = pow((j - KERNEL_SIZE/2), 2);

            double stDevSq = pow(stDev, 2);
            double pi = M_PI;

            //calculate the value at each index of the Kernel
            double kernelVal = exp(-(((xComp) + (yComp)) / (2 * stDevSq)));
            kernelVal = (1 / (sqrt(2 * pi)*stDev)) * kernelVal;

            //populate Kernel
            kernel[i][j] = kernelVal;

            if (i==0 && j==0) 
            {
                scaleVal = kernel[0][0];
            }

            //normalize Kernel
            kernel[i][j] = kernel[i][j] / scaleVal;
        }
    }*/

    int rows = m_image_mgr->getImgHeight();
    int cols = m_image_mgr->getImgWidth();
    double kernelSum;
    double redPixelVal;
    double greenPixelVal;
    double bluePixelVal;

    //Apply Kernel to image
    for (int pixNum = 0; pixNum < rows * cols; ++pixNum) {

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
        out_pixels[pixNum].red = redPixelVal / kernelSum;
        out_pixels[pixNum].green = greenPixelVal / kernelSum;
        out_pixels[pixNum].blue = bluePixelVal / kernelSum;
        redPixelVal = 0;
        greenPixelVal = 0;
        bluePixelVal = 0;
        kernelSum = 0;
    }            
}

///
/// \brief Compute gradient (first order derivative x and y)
///
void CannyEdgeDetector::compute_intensity_gradient(pixel_t *in_pixels, pixel_channel_t_signed *deltaX_channel, pixel_channel_t_signed *deltaY_channel,unsigned max_pixel_cnt)
{
    unsigned offset = m_image_mgr->getImgWidth();
    unsigned parser_length = m_image_mgr->getImgHeight();
    unsigned idx;
    pixel_t_signed *deltaX = new pixel_t_signed[max_pixel_cnt];
    pixel_t_signed *deltaY = new pixel_t_signed[max_pixel_cnt];
    assert(nullptr != deltaX);
    assert(nullptr != deltaY);

    // compute delta X ***************************
    // deltaX = f(x+1) - f(x-1)
    for(unsigned i = 0; i < parser_length; ++i)
    {
        idx = offset * i; // current position X per line

        // gradient at the first pixel of each line
        // note: the edge,pix[idx-1] is NOT exsit
        deltaX[idx].red = (int16_t)(in_pixels[idx+1].red - in_pixels[idx].red);
        deltaX[idx].green = (int16_t)(in_pixels[idx+1].green - in_pixels[idx].green);
        deltaX[idx].blue = (int16_t)(in_pixels[idx+1].blue - in_pixels[idx].blue);

        // gradients where NOT edge
        for(unsigned j = 1; j < offset-1; ++j)
        {
            idx++;
            deltaX[idx].red = (int16_t)(in_pixels[idx+1].red - in_pixels[idx-1].red);
            deltaX[idx].green = (int16_t)(in_pixels[idx+1].green - in_pixels[idx-1].green);
            deltaX[idx].blue = (int16_t)(in_pixels[idx+1].blue - in_pixels[idx-1].blue);
        }

        // gradient at the last pixel of each line
        idx++;
        deltaX[idx].red = (int16_t)(in_pixels[idx].red - in_pixels[idx-1].red);
        deltaX[idx].green = (int16_t)(in_pixels[idx].green - in_pixels[idx-1].green);
        deltaX[idx].blue = (int16_t)(in_pixels[idx].blue - in_pixels[idx-1].blue);
    }

    // compute delta Y ***************************
    // deltaY = f(y+1) - f(y-1)
    for(unsigned j = 0; j < offset; ++j)
    {
        idx = j;    // current Y position per column
        // gradient at the first pixel
        deltaY[idx].red = (int16_t)(in_pixels[idx+offset].red - in_pixels[idx].red);
        deltaY[idx].green = (int16_t)(in_pixels[idx+offset].green - in_pixels[idx].green);
        deltaY[idx].blue = (int16_t)(in_pixels[idx+offset].blue - in_pixels[idx].blue);

        // gradients for NOT edge pixels
        for(unsigned i = 1; i < parser_length-1; ++i)
        {
            idx += offset;
            deltaY[idx].red = (int16_t)(in_pixels[idx+offset].red - in_pixels[idx-offset].red);
            deltaY[idx].green = (int16_t)(in_pixels[idx+offset].green - in_pixels[idx-offset].green);
            deltaY[idx].blue = (int16_t)(in_pixels[idx+offset].blue - in_pixels[idx-offset].blue);
        }

        // gradient at the last pixel of each column
        idx += offset;
        deltaY[idx].red = (int16_t)(in_pixels[idx].red - in_pixels[idx-offset].red);
        deltaY[idx].green = (int16_t)(in_pixels[idx].green - in_pixels[idx-offset].green);
        deltaY[idx].blue = (int16_t)(in_pixels[idx].blue - in_pixels[idx-offset].blue);
    }
    for(idx = 0; idx < max_pixel_cnt; idx++)
    {
        deltaX_channel[idx] = 0.2989 * deltaX[idx].red + 0.5870 * deltaX[idx].green + 0.1140 * deltaX[idx].blue;
        deltaY_channel[idx] = 0.2989 * deltaY[idx].red + 0.5870 * deltaY[idx].green + 0.1140 * deltaY[idx].blue; 
    }
    delete [] deltaX;
    delete [] deltaY;
}


///
/// \brief Compute magnitude of gradient(deltaX & deltaY) per pixel.
///
void CannyEdgeDetector::magnitude(pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *out_pixel, unsigned max_pixel_cnt)
{
    unsigned idx;
    unsigned offset = m_image_mgr->getImgWidth();
    unsigned parser_length = m_image_mgr->getImgHeight();
 
    //computation
    idx = 0;
    for(unsigned i = 0; i < parser_length; ++i)
        for(unsigned j = 0; j < offset; ++j, ++idx)
        {
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
void CannyEdgeDetector::suppress_non_max(pixel_channel_t *mag, pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *nms)
{
    unsigned t = 0;
    unsigned offset = m_image_mgr->getImgWidth();
    unsigned parser_length = m_image_mgr->getImgHeight();
    float alpha;
    float mag1, mag2;
    const pixel_channel_t SUPPRESSED = 0;

    // put zero all boundaries of image
    // TOP edge line of the image
    for(unsigned j = 0; j < offset; ++j)
        nms[j] = 0;

    // BOTTOM edge line of image
    t = (parser_length-1)*offset;
    for(unsigned j = 0; j < offset; ++j, ++t)
        nms[t] = 0;

    // LEFT & RIGHT edge line
    t = offset;
    for(unsigned i = 1; i < parser_length; ++i, t+=offset)
    {
        nms[t] = 0;
        nms[t+offset-1] = 0;
    }

    t = offset + 1;  // skip boundaries of image
    // start and stop 1 pixel inner pixels from boundaries
    for(unsigned i = 1; i < parser_length-1; i++, t+=2)
    {
        for(unsigned j = 1; j < offset-1; j++, t++)
        {
            // if magnitude = 0, no edge
            if(mag[t] == 0) nms[t] = SUPPRESSED;
            else{
                if(deltaX[t] >= 0)
                {
                    if(deltaY[t] >= 0)  // dx >= 0, dy >= 0
                    {
                        if((deltaX[t] - deltaY[t]) >= 0)       // direction 1 (SEE, South-East-East)
                        {
                            alpha = (float)deltaY[t] / deltaX[t];
                            mag1 = (1-alpha)*mag[t+1] + alpha*mag[t+offset+1];
                            mag2 = (1-alpha)*mag[t-1] + alpha*mag[t-offset-1];
                        }
                        else                                // direction 2 (SSE)
                        {
                            alpha = (float)deltaX[t] / deltaY[t];
                            mag1 = (1-alpha)*mag[t+offset] + alpha*mag[t+offset+1];
                            mag2 = (1-alpha)*mag[t-offset] + alpha*mag[t-offset-1];
                        }
                    }

                    else  // dx >= 0, dy < 0
                    {
                        if((deltaX[t] + deltaY[t]) >= 0)    // direction 8 (NEE)
                        {
                            alpha = (float)-deltaY[t] / deltaX[t];
                            mag1 = (1-alpha)*mag[t+1] + alpha*mag[t-offset+1];
                            mag2 = (1-alpha)*mag[t-1] + alpha*mag[t+offset-1];
                        }
                        else                                // direction 7 (NNE)
                        {
                            alpha = (float)deltaX[t] / -deltaY[t];
                            mag1 = (1-alpha)*mag[t+offset] + alpha*mag[t+offset-1];
                            mag2 = (1-alpha)*mag[t-offset] + alpha*mag[t-offset+1];
                        }
                    }
                }

                else
                {
                    if(deltaY[t] >= 0) // dx < 0, dy >= 0
                    {
                        if((deltaX[t] + deltaY[t]) >= 0)    // direction 3 (SSW)
                        {
                            alpha = (float)-deltaX[t] / deltaY[t];
                            mag1 = (1-alpha)*mag[t+offset] + alpha*mag[t+offset-1];
                            mag2 = (1-alpha)*mag[t-offset] + alpha*mag[t-offset+1];
                        }
                        else                                // direction 4 (SWW)
                        {
                            alpha = (float)deltaY[t] / -deltaX[t];
                            mag1 = (1-alpha)*mag[t-1] + alpha*mag[t+offset-1];
                            mag2 = (1-alpha)*mag[t+1] + alpha*mag[t-offset+1];
                        }
                    }

                    else // dx < 0, dy < 0
                    {
                        if((-deltaX[t] + deltaY[t]) >= 0)   // direction 5 (NWW)
                        {
                            alpha = (float)deltaY[t] / deltaX[t];
                            mag1 = (1-alpha)*mag[t-1] + alpha*mag[t-offset-1];
                            mag2 = (1-alpha)*mag[t+1] + alpha*mag[t+offset+1];
                        }
                        else                                // direction 6 (NNW)
                        {
                            alpha = (float)deltaX[t] / deltaY[t];
                            mag1 = (1-alpha)*mag[t-offset] + alpha*mag[t-offset-1];
                            mag2 = (1-alpha)*mag[t+offset] + alpha*mag[t+offset+1];
                        }
                    }
                }

                // non-maximal suppression
                // compare mag1, mag2 and mag[t]
                // if mag[t] is smaller than one of the neighbours then suppress it
                if((mag[t] < mag1) || (mag[t] < mag2))
                    nms[t] = SUPPRESSED;
                else
                {
                    nms[t] = mag[t];
                }

            } // END OF ELSE (mag != 0)
        } // END OF FOR(j)
    } // END OF FOR(i)
}

///
/// \brief Hysteresis step. This is used to 
/// a) remove weak edges 
/// b) connect "split edges" (to preserve weak-touching-strong edges)
///
void CannyEdgeDetector::apply_hysteresis(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, pixel_channel_t t_high, pixel_channel_t t_low)
{
    /* skip first and last rows and columns, since we'll check them as surrounding neighbors of 
     * the adjacent rows and columns */
    unsigned offset = m_image_mgr->getImgWidth();
    unsigned parser_length = m_image_mgr->getImgHeight();
    for(unsigned i = 1; i < parser_length - 1; i++) {
        for(unsigned j = 1; j < offset - 1; j++) {
            unsigned t = (m_image_mgr->getImgWidth() * i) + j;
            /* if our input is above the high threshold and the output hasn't already marked it as an edge */
            if ((in_pixels[t] > t_high) && (out_pixels[t] != m_edge)) {
                /* mark as strong edge */
                out_pixels[t] = m_edge;

                /* check 8 immediately surrounding neighbors 
                 * if any of the neighbors are above the low threshold, preserve edge */
                trace_immed_neighbors(out_pixels, in_pixels, t, t_low);
            }
        }
    }
}

///
/// \brief This function looks at the 8 surrounding neighbor pixels of a given pixel and 
/// marks them as edges if they're above a low threshold value. Used in hysteresis.
///
void CannyEdgeDetector::trace_immed_neighbors(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, unsigned idx, pixel_channel_t t_low)
{
    assert(nullptr != in_pixels);
    assert(nullptr != out_pixels);
    
    /* directions representing indices of neighbors */
    unsigned n, s, e, w;
    unsigned nw, ne, sw, se;

    /* get indices */
    n = idx - m_image_mgr->getImgWidth();
    nw = n - 1;
    ne = n + 1;
    s = idx + m_image_mgr->getImgWidth();
    sw = s - 1;
    se = s + 1;
    w = idx - 1;
    e = idx + 1;

    if ((in_pixels[nw] >= t_low) && (out_pixels[nw] != m_edge)) {
        out_pixels[nw] = m_edge;
    }
    if ((in_pixels[n] >= t_low) && (out_pixels[n] != m_edge)) {
        out_pixels[n] = m_edge;
    }
    if ((in_pixels[ne] >= t_low) && (out_pixels[ne] != m_edge)) {
        out_pixels[ne] = m_edge;
    }
    if ((in_pixels[w] >= t_low) && (out_pixels[w] != m_edge)) {
        out_pixels[w] = m_edge;
    }
    if ((in_pixels[e] >= t_low) && (out_pixels[e] != m_edge)) {
        out_pixels[e] = m_edge;
    }
    if ((in_pixels[sw] >= t_low) && (out_pixels[sw] != m_edge)) {
        out_pixels[sw] = m_edge;
    }
    if ((in_pixels[s] >= t_low) && (out_pixels[s] != m_edge)) {
        out_pixels[s] = m_edge;
    }
    if ((in_pixels[se] >= t_low) && (out_pixels[se] != m_edge)) {
        out_pixels[se] = m_edge;
    }
}
