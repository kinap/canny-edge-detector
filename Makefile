CURDIR := $(shell pwd)
RUNDIR := $(CURDIR)/bin

# Magick++ settings
MAGDIR := $(CURDIR)/third_party/usr
CFG_TOOL := $(MAGDIR)/bin/Magick++-config
PKG_CONFIG_PATH := $(MAGDIR)/lib/pkgconfig
LD_LIBRARY_PATH := $(MAGDIR)/lib
IM_CXXFLAGS := $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) $(CFG_TOOL) --cxxflags --cppflags)
IM_LDFLAGS := $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) $(CFG_TOOL) --ldflags --libs)

# Project settings
PROJ = edge_detect
CC = g++
CFLAGS = $(IM_CXXFLAGS) -std=c++11 -Wall -Werror
DEPOPTS = -std=c++11 -MM
INC = -I$(CURDIR)/inc

#-- Do not edit below this line --

ifeq ($(DEBUG),1)
CFLAGS += -DDEBUG -g -O0
else
CLAGS += -O2
endif

# Subdirs to search for additional source files
SUBDIRS := $(shell ls -F | grep "src" )
DIRS := ./ $(SUBDIRS)
SOURCE_FILES := $(foreach d, $(DIRS), $(wildcard $(d)*.cpp) )

# Objects
OBJS = $(patsubst %.cpp, %.o, $(SOURCE_FILES))

# Dependencies
DEPS = $(patsubst %.cpp, %.d, $(SOURCE_FILES))

# Create .d files
%.d: %.cpp
	$(CC) $(DEPOPTS) $< -MT "$*.o $*.d" -MF $*.d $(INC)

# Make $(PROJ) the default target
all: $(DEPS) $(PROJ)

$(PROJ): $(OBJS)
	$(CC) -o $(RUNDIR)/$(PROJ) $(OBJS) $(INC) $(IM_LDFLAGS)

# Include any dependencies
ifneq "$(strip $(DEPS))" ""
-include $(DEPS)
endif

# Compile every cpp file to an object
%.o: %.cpp
	$(CC) -c $(CFLAGS) -o $@ $< $(INC)

# Clean
.PHONY: clean
clean:
	rm -f $(PROJ)
	rm -f $(OBJS)
	rm -f $(DEPS)

run:
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(RUNDIR)/$(PROJ)
