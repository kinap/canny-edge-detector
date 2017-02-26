
#ifndef _CANNY_EDGE_DETECTOR_HPP_
#define _CANNY_EDGE_DETECTOR_HPP_

#include "edgeDetector.hpp"

///
/// \brief Canny edge dectectior
///
/// Encapsulates the canny edge detection algorithm.
///
class CannyEdgeDetector : public EdgeDetector
{
    public:
        CannyEdgeDetector(std::shared_ptr<ImgMgr> image);
        ~CannyEdgeDetector();

        void detect_edges(bool serial);

    private:
        /* these are all called by detect_edges() to implement the algorithm */
        //void apply_gaussian_filter();
        //void compute_intensity_gradient();
        //void suppress_non_max();
        //void apply_double_threshold();
        void apply_hysteresis(pixel_t *pixels, pixel_t hi_thld, pixel_t lo_thld);
};

#endif // _CANNY_EDGE_DETECTOR_HPP_
