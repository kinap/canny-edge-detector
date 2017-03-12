
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

    /* sanity checks */
    if (0 == args.inFile.compare(args.outFile)) {
        std::cerr << "Input and output file names must be different!" << std::endl;
        exit(ED_PARSE_ERR);
    }

    std::cout << "Canny Edge Detection" << std::endl;
    if (true == args.serial) {
        std::cout << "Executing serially on CPU" << std::endl;
    } else {
        std::cout << "Executing in parallel on GPU" << std::endl;
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
    std::cout << "Edge detection complete" << std::endl;

    return ED_SUCCESS;
}
