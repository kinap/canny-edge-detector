
#include <iostream>
#include "ed_error.h"
#include "ed_args.h"
#include "imgMgr.hpp"
#include "cannyEdgeDetector.hpp"
#include "canny.h"

int main(int argc, char** argv)
{
    /* storage and defaults for command line arguments */
    struct arguments args;
        args.inFile = DEFAULT_INFILE;
        args.outFile = DEFAULT_OUTFILE; 
        args.serial = false;

    /* parse cmd line args */
    int rc = argp_parse(&argp, argc, argv, 0, 0, &args);
    if (rc) {
        std::cerr << "Failed to parse command line arguments." << std::endl;
        exit(rc);
    }

    /* Instantiate our image manager */
    std::shared_ptr<ImgMgr> img_mgr = std::make_shared<ImgMgr>(*argv);

    /* read input file */
    img_mgr->read_image(args.inFile);

    /* Instantiate our edge detector */
    CannyEdgeDetector ced(img_mgr);

    /* run edge detection algorithm */
    ced.detect_edges(args.serial);

    /* write results */
    img_mgr->write_image(args.outFile);

    return ED_SUCCESS;
}
