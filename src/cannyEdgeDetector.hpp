
#ifndef _CANNY_EDGE_DETECTOR_HPP_
#define _CANNY_EDGE_DETECTOR_HPP_

#include "edgeDetector.hpp"

struct pixel_t_signed {
    int16_t red;
    int16_t green;
    int16_t blue;
};

struct pixel_t_float {
    float red;
    float green;
    float blue;
};

typedef int16_t pixel_channel_t_signed;

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
        /* CPU implementation */
        void apply_gaussian_filter(pixel_t *blurred_pixels, pixel_t *input_pixels, unsigned input_pixel_length);
        void compute_intensity_gradient(pixel_t *in_pixels, pixel_t_signed *deltaX, pixel_t_signed *deltaY);
        void suppress_non_max(float *mag, pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, float *nms);
        void apply_hysteresis(pixel_t *out_pixels, pixel_t *in_pixels, pixel_t hi_thld, pixel_t lo_thld);

        void magnitude(pixel_t_signed *deltaX, pixel_t_signed *deltaY, pixel_t_float *mag);
        void rgb2gray(pixel_t_signed *in_pixel, pixel_channel_t_signed *out_pixel, unsigned max_pixel_cnt);
	void rgb2gray_float(pixel_t_float *in_pixel, float *out_pixel, unsigned max_pixel_cnt);
        //void apply_double_threshold();

        /* CUDA/GPU implementation */
        //void cu_apply_gaussian_filter();
        //void cu_compute_intensity_gradient();
        //void cu_suppress_non_max();
        //void cu_apply_double_threshold();
        //void cu_apply_hysteresis(pixel_t *out_pixels, pixel_t *in_pixels, pixel_t hi_thld, pixel_t lo_thld);

        /* helper functions */
        void trace_immed_neighbors(pixel_t *out_pixels, pixel_t *in_pixels, unsigned idx, pixel_t t_low);

        /* member variables */
        pixel_t m_edge; // defines an edge for image this detector was initialized with
};

#endif // _CANNY_EDGE_DETECTOR_HPP_
