# Canny Edge Dectector
Canny edge detector written in CUDA C++.

# Dependencies
- CUDA GPGPU framework
- Magick++ image library

## Magick++ Installation
```
cd third_party
./install_magick++.sh
source magick.env
```

# Compilation
Binaries will be located in the bin/ directory. Ensure all dependencies are installed and that magick.env has been sourced. 

Then run:
```
make
```

# Execution
The edge detector supports serial execution on a host CPU, or parallel execution on an NVIDIA GPU.
For command line argument specifics, use the --help option on the generated binary.

To run with default arguments:
```
make run
```
