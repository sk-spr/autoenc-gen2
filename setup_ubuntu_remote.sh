#!/bin/bash
projectfolder="$(pwd)"
cd $projectfolder/..

# install prerequisites
sudo apt update
sudo apt install build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libncurses-dev tk-dev cmake make clang proj-bin libproj-dev

# build GDAL from source (newer versions not available on LTS Ubuntu)
git clone https://github.com/OSGeo/GDAL
cd $projectfolder/../GDAL
mkdir build
cd build
cmake .
cmake --build . -j16
sudo cmake --build . --target install

# download and convert switzerland data
cd $projectfolder
pip install poetry --break-system-packages
$(poetry env activate)
export DATADIR=data/tiles/
export RAWDIR=data/raw
mkdir -p $DATADIR
mkdir -p $RAWDIR
poetry update && python download_swiss_2m.py 
gdalbuildvrt swit.vrt data/raw/*.tif
gdal raster tile -i swit.vrt -f GTiff -o data/tiles --tiling-scheme WebMercatorQuad --max-zoom 18 --tile-size 256 --skip-blank --resume --parallel-method spawn --no-alpha --excluded-values 0,0,0 --resampling average --co NUM_THREADS=16