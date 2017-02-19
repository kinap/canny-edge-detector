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
Ensure all dependencies are installed first. Binaries will be located in the bin/ directory.
```
make
```

# Execution
The edge detector supports serial execution on a host CPU, or paralell execution on an NVIDIA GPU.

For command line argument specifics, use the --help option on the generated binary.
