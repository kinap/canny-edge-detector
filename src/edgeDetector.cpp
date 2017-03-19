
#include "edgeDetector.hpp"

EdgeDetector::EdgeDetector(std::shared_ptr<ImgMgr> image)
: m_image_mgr(image)
{

}

EdgeDetector::~EdgeDetector(void)
{

}

///
/// \brief Convert single channel to grayscale. 
///
void EdgeDetector::single_channel_to_grayscale(pixel_t *out_grayscale, pixel_channel_t *in_pixels, unsigned rows, unsigned cols)
{
    unsigned idx = 0;
 
    for(unsigned i = 0; i < rows; ++i) {
        for(unsigned j = 0; j < cols; ++j, ++idx) {
            out_grayscale[idx].red = in_pixels[idx];
            out_grayscale[idx].green = in_pixels[idx];
            out_grayscale[idx].blue = in_pixels[idx];
        }
    }
}
