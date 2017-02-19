
#include <iostream>
#include "ced_error.h"
#include "ced_args.h"
#include "imgMgr.hpp"
#include "cannyEdgeDetector.hpp"

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
    ImgMgr img_mgr(*argv);
    img_mgr.test("test.bmp");

    /* Instantiate our edge detector */
    CannyEdgeDetector ced;
    ced.detect_edges(args.serial);

    return CED_SUCCESS;
}
