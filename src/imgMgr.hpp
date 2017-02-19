
#include <string>
#include <Magick++.h>

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

        void test(std::string out_filename);

    private:

};
