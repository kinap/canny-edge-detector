
#include <iostream>
#include "cannyEdgeDetector.hpp"

CannyEdgeDetector::CannyEdgeDetector(std::shared_ptr<ImgMgr> image)
: EdgeDetector(image)
{

}

CannyEdgeDetector::~CannyEdgeDetector(void)
{

}

void CannyEdgeDetector::detect_edges(bool serial)
{
    std::cout << "in canny edge detector" << std::endl;
}
