CURDIR := $(shell pwd)
RUNDIR := $(CURDIR)/bin

# Magick++ settings
MAGDIR := $(CURDIR)/third_party/usr
CFG_TOOL := $(MAGDIR)/bin/Magick++-config
PKG_CONFIG_PATH := $(MAGDIR)/lib/pkgconfig
LD_LIBRARY_PATH := $(MAGDIR)/lib
IM_CXXFLAGS := $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) $(CFG_TOOL) --cxxflags)
IM_LDFLAGS := $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) $(CFG_TOOL) --ldflags)

# Project settings
PROJ = edge_detect
CC = g++
CFLAGS = -std=c++11 -Wall -Werror $(IM_CXXFLAGS) 
DEPOPTS = -std=c++11 -MM
NVCC = nvcc
NVCFLAGS = -std=c++11 -D_FORCE_INLINES
INC = -I$(CURDIR)/inc

#-- Do not edit below this line --

ifeq ($(DEBUG),1)
CFLAGS += -DDEBUG -g -O0
NVCFLAGS += -g -G -O0
else
CLAGS += -O2
NVCFLAGS += -O2
endif

# Subdirs to search for additional source files
SUBDIRS := $(shell ls -F | grep "src" )
DIRS := ./ $(SUBDIRS)
SOURCE_FILES := $(foreach d, $(DIRS), $(wildcard $(d)*.cpp) )
CUDA_FILES := $(foreach d, $(DIRS), $(wildcard $(d)*.cu) )

# Objects - list a .o for every .cpp and .cu
OBJS = $(patsubst %.cpp, %.o, $(SOURCE_FILES))
CUDA_OBJS = $(patsubst %.cu, %.o, $(CUDA_FILES))

# Dependencies - list a .d for every .cpp
DEPS = $(patsubst %.cpp, %.d, $(SOURCE_FILES))

# Default target - create dependecnies, objects, then executable
all: $(DEPS) $(OBJS) $(CUDA_OBJS) $(PROJ)

# We have to link the executable with nvcc to access the CUDA framework
$(PROJ):
	$(NVCC) -o $(RUNDIR)/$(PROJ) $(OBJS) $(CUDA_OBJS) $(INC) $(IM_LDFLAGS)

# Create dependency file for every cpp file
%.d: %.cpp
	$(CC) $(DEPOPTS) $< -MT "$*.o $*.d" -MF $*.d $(INC)

# Compile object file for every cpp file
%.o: %.cpp
	$(CC) -c $(CFLAGS) -o $@ $< $(INC)

# Compile object file for every cuda file
%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $< $(INC)

# Clean
.PHONY: clean
clean:
	rm -f $(PROJ)
	rm -f $(OBJS)
	rm -f $(DEPS)
	rm -f $(CUDA_OBJS)

run:
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(RUNDIR)/$(PROJ) -s -i ./img/Engine_in.PNG -o ./img/Engine_out.PNG
