
#include "imgMgr.hpp"

ImgMgr::ImgMgr(char *argv)
{
    /* Connect to Magick++ image handler */
    Magick::InitializeMagick(argv);
}

ImgMgr::~ImgMgr(void)
{

}

void ImgMgr::test(std::string out_filename)
{
    /* create a white image with a red dot in the center */
    Magick::Image image("1000x1000", "white");
    image.pixelColor(499,499,"red");
    image.write(out_filename.c_str());
}
