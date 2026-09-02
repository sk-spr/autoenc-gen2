# Training Instructions
1. Save repository in a location with 100GB of free space
2. `bash ./download_at_dtm.sh` to download and prepare 5m-resolution GeoTIFF for Austria and set $GEOTIFF_FILE
3. `bash ./create_tiles.sh` to generate individual .tif files for each tile in data/tiles
4. `poetry env use <path-to-python3.13>` Poetry project should include all dependencies, python 3.13.15 used in testing
5. `poetry install` dependency installation
6. `bash train.sh` train with automatically guessed hyperparameters, and start tensorboard
