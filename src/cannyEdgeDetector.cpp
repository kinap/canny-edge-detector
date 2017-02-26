
#include <iostream>
#include "cannyEdgeDetector.hpp"

CannyEdgeDetector::CannyEdgeDetector(std::shared_ptr<ImgMgr> image)
: EdgeDetector(image)
{

}

CannyEdgeDetector::~CannyEdgeDetector(void)
{

}

void CannyEdgeDetector::detect_edges(bool serial)
{
    std::cout << "in canny edge detector" << std::endl;
}

///
/// \brief Hysteresis step. This is used to 
/// a) remove weak edges 
/// b) connect "split edges" (to preserve weak-touching-strong edges)
///
void CannyEdgeDetector::apply_hysteresis(pixel_t *pixels, pixel_t hi_thld, pixel_t lo_thld)
{
    /* these loops are good candidates for GPU parallelization */
    for (unsigned i = 0; i < m_image_mgr->getImgWidth(); i++) {
        for (unsigned j = 0; j < m_image_mgr->getImgHeight(); j++) {
            unsigned idx = m_image_mgr->getImgWidth() * i + j;
            // TODO which channel should I use here?
            if (pixels[idx].red > hi_thld.red) {
                /* check 8 immediately surrounding neighbors */
                /* if any of the neighbors are above the low threshold, preserve edge */
                // visit_immed_neighbors(pixels[idx], low_threshold);
            }
        }
    }
}
