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
    
    with open("data/swisstopo_links.csv", "r") as link_list:
        links = link_list.readlines()
        links = list(map(lambda x: x.replace("\n",""), links))
        if len(glob.glob(f"{raw_folder}/*.tif")) < len(links):
            print("downloading...")
            print("There are {} CPUs on this machine ".format(cpu_count()))
            pool = Pool(cpu_count() * 16)
            download_func = partial(download_tile, raw_folder)
            results = pool.map(download_func, links)
            pool.close()
            pool.join()
        else:
            print("Already have all files")
    print("bye!")
