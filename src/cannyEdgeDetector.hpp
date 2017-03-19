
#ifndef _CANNY_EDGE_DETECTOR_HPP_
#define _CANNY_EDGE_DETECTOR_HPP_
#define KERNEL_SIZE 7
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
        /* CPU implementation */
        void apply_gaussian_filter(pixel_t *blurred_pixels, pixel_t *input_pixels, double kernel[KERNEL_SIZE][KERNEL_SIZE]);
        void compute_intensity_gradient(pixel_t *in_pixels, pixel_channel_t_signed *deltaX_channel, pixel_channel_t_signed *deltaY_channel,unsigned max_pixel_cnt);
        void magnitude(pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *out_pixel, unsigned max_pixel_cnt);
        void suppress_non_max(pixel_channel_t *mag, pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *nms);
        void apply_hysteresis(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, pixel_channel_t hi_thld, pixel_channel_t lo_thld);


        /* helper functions */
        void populate_blur_kernel(double out_kernel[KERNEL_SIZE][KERNEL_SIZE]);
        void trace_immed_neighbors(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, unsigned idx, pixel_channel_t t_low);

        /* member variables */
        pixel_channel_t m_edge; // defines an edge for image this detector was initialized with
};

#endif // _CANNY_EDGE_DETECTOR_HPP_
