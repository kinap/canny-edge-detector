# Canny Edge Detector
Canny edge detector written in CUDA C++. Provides both serial and GPU implementations.

# Getting Started
This guide assumes you're on a Linux system with the NVIDIA toolchain supporting CUDA (nvcc, nsight, etc.) already installed.
1. Install Magick++ library (see below, just need to run a script and wait for installation to complete)
2. Source the magick.env script (see below)
3. Compile the edge_detect program (see below)
4. Run edge_detect --help to see command line arguments. The user can specify input and output filenames, as well as execution of the serial (CPU) or parallel (GPU) implementation.
5. Run the edge_detect program on an image of your choice (sample images provided in img/)

# Dependencies
- Linux
- CUDA GPGPU framework (nvcc)
- Magick++ image library

## Magick++ Installation
The edge detection program uses the Magick++ library to read and write images of arbitrary encoding. The library also provides low-level access to the raw pixels of an image, which are then transformed and filtered to produce an edge detected image.

Note: we assume that the user does not have administrative access, and thus can't install packages (as might be the case on a shared server). For that reason, we download and install Magick++7 in-tree in a non-standard location that the user has full control over.
```
./third_party/install_magick++.sh
source ./third_party/magick.env
```

# Compilation
Binaries will be located in the bin/ directory. All CPU code objects are compiled with g++. CUDA kernel objects are compiled with nvcc. The final executable is also compiled with nvcc.
Note that multithreaded make is broken with nvcc in this build system.

Ensure all dependencies have installed successfully, then run:
```
make
```

# Execution
The edge detector supports serial execution on a host CPU, or parallel execution on an NVIDIA GPU.
For command line argument specifics, use the --help option on the generated binary.

To run with default arguments (serial implementation, engine test image):
```
make run
```

If you plan to run the binary directly (i.e. not using the makefile), ensure you source magick.env:
```
source third_party/magick.env
```

# Debug
To debug, you'll need to compile with the debug option and run manually.
```
# source environment for manual cmd line execution
source third_party/magick.env

# remove old binaries
make clean

# compile with debug symbols and statements
make DEBUG=1
```

## Debugging Host code

```
# run GNU Debugger
gdb ./bin/edge_detect

# once in the gdb terminal, set breakpoints, arguments, etc.
> set args <cmd line args>
> b <file.extension>:<line number> # set break point
> run # run program until breakpoint
> n # execute next line
> step # step down into next line (inside functions, etc.)
```

## Debugging GPU code
Look for online tutorials for this, it's a bit more involved, though similar, to regular gdb-based debugging. This can also be done using the Nsight tool if you have Eclipse installed.

```
# run CUDA-GNU Debugger
cuda-gdb
```
