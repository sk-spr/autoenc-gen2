# Training Instructions
## Prerequisites:
- for data acquisition: gdal (system), poetry-configured dependencies
- for training: poetry-configured dependencies

1. Save repository in a location with 140GB of free space
2. In project directory, `poetry env use <path-to-python3.13>`
3. In project directory, `$(poetry env activate)`
4. `poetry update`
5. `mkdir -p data/raw data/tiles` to create folders for raw and tiled data
6. To download 1024*1024 GTiff tiles of switzerland at 50cm `RAWDIR=./data/raw python download_swiss_2m.py`
   * Files are ca. 20 MB each, adjust to storage availability as you'll need at least twice as much space. Drop lines from the CSV file if needed.
7. `bash ./create_tiles.sh` to generate individual .tif files for each tile in data/tiles
8. `gdalbuildvrt swit.vrt data/raw/*.tif` build VRT to retile from
9. For train-ready tiles in data/tiles, run `gdal raster tile -i swit.vrt -f GTiff -o data/tiles --tiling-scheme WebMercatorQuad --max-zoom 18 --tile-size 256 --skip-blank --resume --parallel-method spawn --no-alpha --excluded-values 0,0,0 --resampling average`
10. When all zoom-level tiles are successfully exported, data/raw can be deleted or reused
11. Ensure tile_folders in sendable.py points to data/tiles
12. Set appropriate metaparameters for data_loader.num_workers and batch_size (through experiment to try and maximize time spent with high utilization)
   * num_workers must be less than or equal to the logical CPU count
   * good guess for 8GB VRAM might be batch_size=64 num_workers=10
   * reasonable guess for 32GB is batch_size=512 num_workers=8
   * etc.
13. `mkdir -p ./models ./runs`
14. `tensorflow --logdir runs --bind_all &` Start tensorboard to monitor loss and intermediate results at port 6006 (HTTP)
15. `python sendable.py` to begin training. Observe in PyCharm plot window or via web browser at port 6006 once training begins.