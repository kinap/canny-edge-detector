#!/bin/bash

TOP=`pwd`/`dirname $BASH_SOURCE`
mkdir -p $TOP/src
mkdir -p $TOP/usr

# download library
cd $TOP/src
curl -L https://www.imagemagick.org/download/ImageMagick.tar.gz | tar zx

# configure
cd $(ls -d */ | head -n 1)
./configure --prefix=$TOP/usr

# build
make 

# install
make install

# make sure it works
make check
