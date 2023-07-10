#!/bin/bash

TOP=`pwd`/`dirname $BASH_SOURCE`
mkdir -p $TOP/src
mkdir -p $TOP/usr

# download library
cd $TOP/src
curl -L https://www.imagemagick.org/download/ImageMagick.tar.gz | tar zx
# curl -L https://github.com/ImageMagick/ImageMagick6/archive/refs/tags/6.9.12-90.tar.gz | tar zx

# configure
cd $(ls -d */ | head -n 1)
./configure --prefix=$TOP/usr --with-png=yes --with-jpeg=yes --with-jxl=yes --with-modules --with-lcms --with-xml --without-x

# build
make 

# install
make install

# make sure it works
make check
