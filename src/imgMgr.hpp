
#include <string>
#include <stdint.h>
#include <Magick++.h>

typedef struct {
    uint16_t red;
    uint16_t green;
    uint16_t blue;
    uint16_t alpha;
} pixel_t;

///
/// \brief Image manager class
///
/// Wrapper for image wrangling library, in our case libmagick++
/// Provides utilites to open and close image files, and provides 
/// the edge detector class with access to the image data.
///
class ImgMgr
{
    public:
        ImgMgr(char *argv);
        ~ImgMgr();

        unsigned getPixelCount();
        void read_image(const std::string &in_filename);
        void write_image(const std::string &out_filename);

        void test(const std::string &out_filename);

    private:
        int m_img_width;
        int m_img_height;
        pixel_t *m_pixels;
};
