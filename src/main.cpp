
#include <Magick++.h>
#include "error_codes.h"

int main(int argc, char** argv)
{
    Magick::InitializeMagick(*argv);

    Magick::Image image("100x100", "white");
    image.pixelColor(49,49,"red");
    image.write("test.bmp");
    
    return CED_SUCCESS;
}
