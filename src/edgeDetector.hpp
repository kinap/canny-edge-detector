
//#include "ImgMgr.hpp"

///
/// \brief Edge dectector class
///
/// Defines the edge detection interface.
/// Interfaces with a wrapper for an image handling library
/// and detects edges based on an arbitrary algorithm.
/// Can run in parallel on a GPU or serially on the host CPU.
///
class EdgeDetector
{
    public:
        EdgeDetector();
        ~EdgeDetector();

        //import_image();

        virtual void detect_edges(bool serial) = 0;

        //export_image(std::string filename);

    private:
};
