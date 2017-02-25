
#include <iostream>
#include "imgMgr.hpp"

ImgMgr::ImgMgr(char *argv)
: m_pixels(nullptr)
{
    /* Initialize our image library */
    Magick::InitializeMagick(argv);
}

ImgMgr::~ImgMgr(void)
{
    if (m_pixels != nullptr) {
        delete [] m_pixels;
    }
}

unsigned ImgMgr::getPixelCount(void)
{
    return m_img_width * m_img_height;
}

void ImgMgr::read_image(const std::string &in_filename)
{
    try {
        Magick::Image img;
        img.read(in_filename);

        /* populate internal data structures */
        m_img_width = img.rows();
        m_img_height = img.columns();

        m_pixels = new pixel_t[getPixelCount()]; // free'd in destructor
        
        /* extract the pixels from the image, put them in a format we can export portably */
        Magick::Pixels view(img);
        /* Magic::Pixels.get() returns an array of pixel channels of type Quantum */
        Magick::Quantum *pixels = view.get(0, 0, img.columns(), img.rows());
        for (unsigned i = 0; i < img.rows(); i++) {
            for (unsigned j = 0; j < img.columns(); j++) {
                unsigned idx = i * img.columns() + j; // remap to a flat buffer of pixel structs
                m_pixels[idx].red = *pixels++;
                m_pixels[idx].green = *pixels++;
                m_pixels[idx].blue = *pixels++;
                // Alpha isn't returned from view.get(), do we need it? TODO
            }
        }
    }

    catch (const Magick::Exception &e) {
        std::cerr << "Error reading image: " << e.what() << std::endl;
    }
}

void ImgMgr::write_image(const std::string &out_filename)
{
    try {
        /* create a new white image of the same size as our input image */
        Magick::Image img(Magick::Geometry(m_img_width, m_img_height), Magick::Color("white"));
        Magick::Pixels view(img);
        Magick::Quantum *pixels = view.get(0, 0, img.columns(), img.rows());
        for (unsigned i = 0; i < img.rows(); i++) {
            for (unsigned j = 0; j < img.columns(); j++) {
                /* extract channels from our array of pixel structs */
                unsigned idx = i * img.columns() + j; 
                *pixels++ = m_pixels[idx].red;
                *pixels++ = m_pixels[idx].green;
                *pixels++ = m_pixels[idx].blue;
            }
        }

        img.write(out_filename);
        
    }

    catch (const Magick::Exception &e) {
        std::cerr << "Error writing image: " << e.what() << std::endl;
    }
}

void ImgMgr::test(const std::string &out_filename)
{
    /* create a white image with a red dot in the center */
    Magick::Image image("1000x1000", "white");
    image.pixelColor(499,499,"red");
    image.write(out_filename.c_str());
}
