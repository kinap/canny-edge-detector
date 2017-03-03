
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

    /* test */
    for (unsigned i = 0; i < input_pixel_length; i++) {
        std::cout << "Pixels[" << std::dec << i << "].red : " << std::hex << (unsigned) raw_pixels[i].red << std::endl;
    }

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
