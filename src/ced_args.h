/* This file contains argument parsing functions and such */

#include <argp.h>
#include <string.h>

#define DEFAULT_INFILE "test.bmp"
#define DEFAULT_OUTFILE "test_out.bmp"

const char *argp_program_version = "edge_detect v0.1.0";

static char doc[] =
"Canny edge detector written in C++. \n\
	Takes an input bitmap (.bmp) image and outputs a bitmap image \n\
	containing strong edges only. \n\n\
	Supports serial and parallel execution of the algorithm. \n\
	Serial version runs on the host CPU, while the parallel version \n\
	supports execution on an NVIDIA GPU by using the CUDA C++ framework.";

/* Options and their descriptions */
static struct argp_option options[] = {
    {"input-file", 'i', "FILENAME", 0, "Input image filename.", 0},
    {"output-file", 'o', "FILENAME", 0, "Output image filename.", 0},
    {"serialize", 's', 0, 0, "Execute serially on host CPU (GPU otherwise).", 0},
    {0, 0, 0, 0, 0, 0}
};

/* struct holding options */
struct arguments {
    std::string inFile;
    std::string outFile;
    bool gpgpu;
};

/* Parser */
static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
    struct arguments *args = (arguments *)state->input;

    switch(key) 
    {
        case 'i':
            args->inFile = arg;
            break;
        case 'o':
            args->outFile = arg;
            break;
        case 's':
            args->gpgpu = false;
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

/* argp interface */
static struct argp argp = {options, parse_opt, NULL, doc, 0, 0, 0};

