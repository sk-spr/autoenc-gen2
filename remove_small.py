import os, glob, random
import shutil

tile_folders = ["/home/skye/data/data/swit_tiles1/18/**/*.tif","/run/media/skye/backup31/dat/switzerland_tiles/18/**/*.tif","/run/media/skye/16769757193425BC/data/swit_tiles1/18/**/*.tif"]

zoom_level = "**"
data_files = []
for glob_expr in tile_folders:
    data_files += glob.glob(glob_expr)
print(f"{len(data_files)} raw files")

remove_files = list(filter(lambda x: os.path.getsize(x) < 2000, data_files))
print(f"Removing {len(remove_files)} files, OK?")
if input("Y/N: ") == "Y":
    for f in remove_files:
        os.remove(f)