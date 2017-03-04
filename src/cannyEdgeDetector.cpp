
#include <iostream> // cout, cerr
#include <assert.h> // assert
#include "string.h" // memcpy
#include "cannyEdgeDetector.hpp"

CannyEdgeDetector::CannyEdgeDetector(std::shared_ptr<ImgMgr> image)
: EdgeDetector(image)
{
    /* a strong edge is the largest value a channel can hold */
    /* e.g. 8 bit channel: (1 << 8) - 1 -> b1_0000_0000 - b1 -> 0_1111_1111 */
    unsigned max_val = (1 << image->getChannelDepth()) - 1;
    m_edge.red = max_val;
}

CannyEdgeDetector::~CannyEdgeDetector(void)
{

}

void CannyEdgeDetector::detect_edges(bool serial)
{
    std::cout << "in canny edge detector" << std::endl;
    pixel_t *raw_pixels = m_image_mgr->getPixelHandle();
    unsigned input_pixel_length = m_image_mgr->getPixelCount();

    if (true == serial) {
        std::cout << "  executing serially" << std::endl;
        pixel_t *buf0 = new pixel_t[input_pixel_length];
        pixel_t *buf1 = new pixel_t[input_pixel_length];
        assert(buf0);
        assert(buf1);

        apply_gaussian_filter(buf0, raw_pixels);
        //compute_intensity_gradient(buf1, buf0);
        //suppress_non_max();
        //apply_double_threshold();
        //apply_hysteresis(pixel_t *out_pixels, pixel_t *in_pixels, pixel_t hi_thld, pixel_t lo_thld);

        memcpy(raw_pixels, buf0, input_pixel_length * sizeof(pixel_t[0]));

        delete [] buf0;
        delete [] buf1;

    } else { // GPGPU
        /* Copy pixels to device - results of each stage stored on GPU and passed to next kernel */
        //cu_apply_gaussian_filter();
        //cu_compute_intensity_gradient();
        //cu_suppress_non_max();
        //cu_apply_double_threshold();
        //cu_apply_hysteresis(pixel_t *out_pixels, pixel_t *in_pixels, pixel_t hi_thld, pixel_t lo_thld);
    }
}

void CannyEdgeDetector::apply_gaussian_filter(pixel_t *blurred_pixels, pixel_t *input_pixels)
{
    std::cout << "heya" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
// compute gradient (first order derivative x and y)
///////////////////////////////////////////////////////////////////////////////
void CannyEdgeDetector::gradient(pixel_t *in_pixels, pixel_t *deltaX, pixel_t *deltaY)
{
	unsigned offset = m_image_mgr->getImgWidth();
	unsigned parser_length = m_image_mgr->getImgHeight();
	unsigned idx;
    // compute delta X ***************************
    // deltaX = f(x+1) - f(x-1)
    for(unsigned i = 0; parser_length; ++i)
    {
		idx = offset * i; // current position X per line

        // gradient at the first pixel of each line
        // note: the edge,pix[idx-1] is NOT exsit
        deltaX[idx].red = in_pixels[idx+1].red - in_pixels[idx].red;
		deltaX[idx].green = in_pixels[idx+1].green - in_pixels[idx].green;
		deltaX[idx].blue = in_pixels[idx+1].blue - in_pixels[idx].blue;

        // gradients where NOT edge
        for(unsigned j = 1; j < offset-1; ++j)
        {
            idx++;
			deltaX[idx].red = in_pixels[idx+1].red - in_pixels[idx-1].red;
			deltaX[idx].green = in_pixels[idx+1].green - in_pixels[idx-1].green;
			deltaX[idx].blue = in_pixels[idx+1].blue - in_pixels[idx-1].blue;
        }

        // gradient at the last pixel of each line
        idx++;
		deltaX[idx].red = in_pixels[idx].red - in_pixels[idx-1].red;
		deltaX[idx].green = in_pixels[idx].green - in_pixels[idx-1].green;
		deltaX[idx].blue = in_pixels[idx].blue - in_pixels[idx-1].blue;
    }

    // compute delta Y ***************************
    // deltaY = f(y+1) - f(y-1)
    for(unsigned j = 0; j < offset; ++j)
    {
		idx = j;	// current Y position per column
        // gradient at the first pixel
		deltaY[idx].red = in_pixels[idx+offset].red - in_pixels[idx].red;
		deltaY[idx].green = in_pixels[idx+offset].green - in_pixels[idx].green;
		deltaY[idx].blue = in_pixels[idx+offset].blue - in_pixels[idx].blue;

        // gradients for NOT edge pixels
        for(unsigned i = 1; i < parser_length-1; ++i)
        {
			idx += offset;
			deltaY[idx].red = in_pixels[idx+offset].red - in_pixels[idx-offset].red;
			deltaY[idx].green = in_pixels[idx+offset].green - in_pixels[idx-offset].green;
			deltaY[idx].blue = in_pixels[idx+offset].blue - in_pixels[idx-offset].blue;
        }

        // gradient at the last pixel of each column
		idx += offset;
		deltaY[idx].red = in_pixels[idx].red - in_pixels[idx-offset].red;
		deltaY[idx].green = in_pixels[idx].green - in_pixels[idx-offset].green;
		deltaY[idx].blue = in_pixels[idx].blue - in_pixels[idx-offset].blue;
    }
}



///////////////////////////////////////////////////////////////////////////////
// compute magnitude of gradient(deltaX & deltaY) per pixel
///////////////////////////////////////////////////////////////////////////////
void CannyEdgeDetector::magnitude(pixel_t *deltaX, pixel_t *deltaY, pixel_t *mag)
{
    unsigned idx;
	unsigned offset = m_image_mgr->getImgWidth();
	unsigned parser_length = m_image_mgr->getImgHeight();

	//computation
    idx = 0;
    for(unsigned i = 0; i < parser_length; ++i)
        for(unsigned j = 0; j < offset; ++j, ++idx)
		{
			mag[idx].red = (unsigned short)(sqrt((double)deltaX[idx].red*deltaX[idx].red + (double)deltaY[idx].red*deltaY[idx].red) + 0.5);
			mag[idx].green = (unsigned short)(sqrt((double)deltaX[idx].green*deltaX[idx].green + (double)deltaY[idx].green*deltaY[idx].green) + 0.5);
			mag[idx].blue = (unsigned short)(sqrt((double)deltaX[idx].blue*deltaX[idx].blue + (double)deltaY[idx].blue*deltaY[idx].blue) + 0.5);
		}
}

void rgb2gray(pixel_t *in_pixel, short *out_pixel, unsigned max_pixel_cnt)
{
	for(unsigned idx = 0; idx < max_pixel_cnt; idx++)
	{
		out_pixel[idx] = 0.2989 * in_pixel[idx].red + 0.5870 * in_pixel[idx].green + 0.1140 * in_pixel[idx].blue; 
	}
	
}

///////////////////////////////////////////////////////////////////////////////
// compute direction of edges for each pixel (angle of 1st derivative of image)
// quantize the normal directions into 16 plus 0(dx=dy=0)
///////////////////////////////////////////////////////////////////////////////
void CannyEdgeDetector::direction(short *deltaX, short *deltaY, unsigned char *orient)
{
    unsigned t = 0;
	unsigned offset = m_image_mgr->getImgWidth();
	unsigned parser_length = m_image_mgr->getImgHeight();
	
    for(unsigned j = 0; j < parser_length; j++)
    {
        for(unsigned i = 0; i < offset; i++)
        {
            if(deltaX[t] == 0) // all axis directions
            {
                if(deltaY[t] == 0) orient[t] = 0;
                else if(deltaY[t] > 0) orient[t] = 5;
                else orient[t] = 13;
            }

            else if(deltaX[t] > 0)
            {
                if(deltaY[t] == 0) orient[t] = 1;
                else if(deltaY[t] > 0)
                {
                    if(deltaX[t] - deltaY[t] == 0) orient[t] = 3;
                    else if(deltaX[t] - deltaY[t] > 0) orient[t] = 2;
                    else orient[t] = 4;
                }
                else
                {
                    if(deltaX[t] + deltaY[t] == 0) orient[t] = 15;
                    else if(deltaX[t] + deltaY[t] > 0) orient[t] = 16;
                    else orient[t] = 14;
                }
            }

            else
            {
                if(deltaY[t] == 0) orient[t] = 9;
                else if(deltaY[t] > 0)
                {
                    if(deltaY[t] + deltaX[t] == 0) orient[t] = 7;
                    else if(deltaY[t] + deltaX[t] > 0) orient[t] = 6;
                    else orient[t] = 8;
                }
                else
                {
                    if(deltaY[t] - deltaX[t] == 0) orient[t] = 11;
                    else if(deltaY[t] - deltaX[t] > 0) orient[t] = 10;
                    else orient[t] = 12;
                }
            }

            //if(orient[t] == 16) printf("%d,%d, %d    ", deltaX[t], deltaY[t],orient[t]);
            t++;
        }
    }
}



///////////////////////////////////////////////////////////////////////////////
// Non Maximal Suppression
// If the centre pixel is not greater than neighboured pixels in the direction,
// then the center pixel is set to zero.
// This process results in one pixel wide ridges.
///////////////////////////////////////////////////////////////////////////////
void CannyEdgeDetector::nonMaxSupp(unsigned short *mag, short *deltaX, short *deltaY, unsigned char *nms)
{
    unsigned t = 0;
	unsigned offset = m_image_mgr->getImgWidth();
	unsigned parser_length = m_image_mgr->getImgHeight();
    float alpha;
    float mag1, mag2;
    const unsigned char SUPPRESSED = 0;

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
                    if(mag[t] > 255) nms[t] = 255;
                    else nms[t] = (unsigned char)mag[t];
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
/// These loops are good candidates for GPU parallelization
///
void CannyEdgeDetector::apply_hysteresis(pixel_t *out_pixels, pixel_t *in_pixels, pixel_t t_high, pixel_t t_low)
{
    /* skip first and last rows and columns, since we'll check them as surrounding neighbors of 
     * the adjacent rows and columns */
    for (unsigned i = 1; i < m_image_mgr->getImgWidth() - 1; i++) {
        for (unsigned j = 1; j < m_image_mgr->getImgHeight() - 1; j++) {
            unsigned idx = m_image_mgr->getImgWidth() * i + j;
            /* if our input is above the high threshold and the output hasn't already marked it as an edge */
            if ((in_pixels[idx] > t_high) && (out_pixels[idx] != m_edge)) {
                /* mark as strong edge */
                out_pixels[idx] = m_edge;

                /* check 8 immediately surrounding neighbors 
                 * if any of the neighbors are above the low threshold, preserve edge */
                trace_immed_neighbors(out_pixels, in_pixels, idx, t_low);
            }
        }
    }
}

///
/// \brief This function looks at the 8 surrounding neighbor pixels of a given pixel and 
/// marks them as edges if they're above a low threshold value. Used in hysteresis.
///
void CannyEdgeDetector::trace_immed_neighbors(pixel_t *out_pixels, pixel_t *in_pixels, unsigned idx, pixel_t t_low)
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
