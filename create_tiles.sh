#!/bin/bash
if [ -f "$GEOTIFF_FILE" ]; then
  gdal raster tile -i $GEOTIFF_FILE -f GTiff ./data/tiles/ --tiling-scheme WebMercatorQuad --max-zoom 15 --tile-size 256 --skip-blank --resume --parallel-method spawn --no-alpha
else
  echo "GEOTIFF_FILE must be specified as variable (WebMercator, GeoTiff)"
fi
