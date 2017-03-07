
#ifndef _EDGE_DETECTOR_HPP_
#define _EDGE_DETECTOR_HPP_

#include "ed_pixel.h"
#include "imgMgr.hpp"

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
        EdgeDetector(std::shared_ptr<ImgMgr> image);
        ~EdgeDetector();

        virtual void detect_edges(bool serial) = 0;

    protected:
        std::shared_ptr<ImgMgr> m_image_mgr;
};

#endif // _EDGE_DETECTOR_HPP_
