#!/bin/bash

TOP=`pwd`
mkdir -p $TOP/src
mkdir -p $TOP/usr

# download library
cd src
curl -L https://www.imagemagick.org/download/ImageMagick.tar.gz | tar zx

# configure
cd ImageMagick-7.0.4-10
./configure --prefix=$TOP/usr

# build
make 

# install
make install

# make sure it works
make check
