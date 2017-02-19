
#include "edgeDetector.hpp"

///
/// \brief Canny edge dectectior
///
/// Encapsulates the canny edge detection algorithm.
///
class CannyEdgeDetector : public EdgeDetector
{
    public:
        CannyEdgeDetector();
        ~CannyEdgeDetector();

        void detect_edges(bool serial);

    private:
        /* these are all called by detect_edges() to implement the algorithm */
        //void apply_gaussian_filter();
        //void compute_intensity_gradient();
        //void suppress_non_max();
        //void apply_double_threshold();
        //void apply_hysteresis();
};
