
#include <iostream>
#include <Magick++.h>
#include "ced_error.h"
#include "ced_args.h"
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

    CannyEdgeDetector ced;
    ced.detect_edges(args.serial);

    /* Connect to Magick++ image handler */
    Magick::InitializeMagick(*argv);

    /* test code */
    Magick::Image image("100x100", "white");
    image.pixelColor(49,49,"red");
    image.write("test.bmp");
    
    return CED_SUCCESS;
}
