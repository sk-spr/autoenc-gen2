#!/bin/bash
temp_file=./data/DGM_R5.tif
echo "downloading 5m heightmap TIFF"
cd data && wget https://data.bev.gv.at/download/DGM/Hoehenraster/DGM_R5.tif --show-progress && cd .. || exit 1
dest_file=./data/DGM_R5_WebMercator.tif
echo "reprojecting..."
gdal raster reproject $temp_file $dest_file -s EPSG:3035 -d EPSG:3857 && rm $temp_file || exit 2
export GEOTIFF_FILE=$dest_file
echo "done!"
