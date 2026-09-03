import glob
import subprocess

import wget
import shutil

import sys, os
import zipfile
import requests
from multiprocessing import Pool, cpu_count
from functools import partial
from io import BytesIO


def download_tile(datafolder,url):
    for i in range(10):
        try:
            file_name = url.split("/")[-1]
            if os.path.exists(datafolder + "/" + file_name):
                return
            response = requests.get(url)
            with open(f"{datafolder}/{file_name}", "wb") as tilefile:
                tilefile.write(response.content)
                tilefile.flush()
                tilefile.close()
            print(f"downloaded {file_name}")
            return
        except Exception as e:
            print(e)
            print(f"retrying {file_name}")


if __name__ == "__main__":
    env = os.environ.copy()
    data_folder = env.get("DATADIR")
    raw_folder = env.get("RAWDIR")
    print(data_folder, raw_folder)
    if len(glob.glob(f"{raw_folder}/*.tif")) < 100:
        print("downloading...")
        with open("data/swisstopo_links.csv", "r") as link_list:
            links = link_list.readlines()
            links = list(map(lambda x: x.replace("\n",""), links))
            print("There are {} CPUs on this machine ".format(cpu_count()))
            pool = Pool(cpu_count())
            download_func = partial(download_tile, raw_folder)
            results = pool.map(download_func, links)
            pool.close()
            pool.join()
    if not os.path.exists(f"{data_folder}/swit_2m.tif"):
        subprocess.run(f"gdal raster mosaic --hide-nodata --output-nodata=0 --output-format COG '{raw_folder}/*.tif' {data_folder}/swit_2m.tif --co BIGTIFF=YES")

    if not os.path.exists(f"{data_folder}/swit_2m_proj.tif"):
        subprocess.run(f"gdal raster reproject {data_folder}/swit_2m.tif {data_folder}/swit_2m_proj.tif -s EPSG:2056 -d EPSG:3857")

    if not os.path.exists(f"{data_folder}/switzerland_tiles"):
        os.mkdir(f"{data_folder}/switzerland_tiles")

        subprocess.run(f"gdal raster tile -i {data_folder}/swit_2m_proj.tif -f GTiff -o {data_folder}/switzerland_tiles --tiling-scheme WebMercatorQuad --max-zoom 18 --tile-size 256 --skip-blank --resume --parallel-method spawn --no-alpha".split(" "))

    print("done, delete unprojected TIFF? (y/N)")
    if input("Type \"y\" to confirm, \"n\" to deny: >") == "y":
        os.remove(f"{data_folder}/swit_2m.tif")
    print("done, delete projected TIFF? (y/N)")
    if input("Type \"y\" to confirm, \"n\" to deny: >") == "y":
        os.remove(f"{data_folder}/swit_2m_proj.tif")
