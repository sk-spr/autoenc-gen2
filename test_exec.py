import glob
import multiprocessing
import os
import random
import time
from functools import partial
from multiprocessing.spawn import freeze_support
from os import PathLike

import torch.export as te
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, BatchSampler
from sendable import EncNet
import onnxruntime

def load_tile(fname: PathLike):
    im = (np.array(Image.open(fname).get_flattened_data()).reshape(1, 256, 256)).astype(np.float32)
    min_val = np.amin(im).astype(np.float32)
    max_val = np.amax(im).astype(np.float32)
    #print(max_val - min_val)
    return (im - min_val) / (max(max_val - min_val, 1))

unloader = transforms.ToPILImage()  # reconvert into PIL image

def get_im(tensor: torch.Tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    return unloader(image)


class HeightTileDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, device, transform=None):
        self.images = len(file_list)
        self.image_files=file_list
        self.transform = transform
        self.device = device
    def __len__(self):
        return self.images
    def __getitem__(self, index):
        imdat = load_tile(self.image_files[index], self.device)
        if self.transform:
            return self.transform(imdat)
        return imdat

if __name__ == "__main__":
    freeze_support()
    torch.multiprocessing.set_start_method('spawn')
    module = torch.load("models/height9_ep0.pt2", weights_only=False)

    torch.onnx.export(
        model=module.encoder,
        args=(torch.randn(128, 1, 256, 256, dtype=torch.float32).cuda(),),
        f="models/height9_encoder_latest.onnx",
        input_names=["image"],
        output_names=["latent"],
        dynamo=True,
        external_data=False,
    )
    torch.onnx.export(
        model=module.decoder,
        args=(torch.randn(128, 2048, dtype=torch.float32).cuda(),),
        f="models/height9_decoder_latest.onnx",
        input_names=["latent"],
        output_names=["image"],
        dynamo=True,
        external_data=False,
    )

    onnx_session_enc = onnxruntime.InferenceSession("models/height8_encoder_latest.onnx", providers=['CUDAExecutionProvider'])
    onnx_session_dec = onnxruntime.InferenceSession("models/height8_decoder_latest.onnx", providers=['CUDAExecutionProvider'])

    data_files = glob.glob("/home/skye/data/switzerland_tiles/18/*/*.tif")
    data_files = list(filter(lambda f: os.path.getsize(f) > 2000,data_files))

    batch_size = 128


    times = []
    for batch_num in range(512):
        random.shuffle(data_files)
        used_files = data_files[:batch_size]
        loaded_files = np.array(list(map(lambda fname: load_tile(fname), used_files)))
        #print(loaded_files.shape)
        time_start = time.time_ns()
        encoded = onnx_session_enc.run(["latent"], {"image": loaded_files})[0]
        time_after_enc = time.time_ns()
        decoded = onnx_session_dec.run(["image"], {"latent": encoded})[0]
        time_after_dec = time.time_ns()

        num_warmup_batch = 16
        if batch_num < num_warmup_batch:
            print(f"Warmup batch {batch_num}: encoding {(((time_after_enc - time_start) / batch_size) / 1000000):<5f} ms; decoding {(((time_after_dec - time_after_enc) / batch_size) / 1000000):<5f}")
        else:
            print(f"Warmed-up batch {batch_num - num_warmup_batch}: encoding {(((time_after_enc - time_start) / batch_size) / 1000000):<5f} ms; decoding {(((time_after_dec - time_after_enc) / batch_size) / 1000000):<5f}")
            times.append(((((time_after_enc - time_start) / batch_size) / 1000000),(((time_after_dec - time_after_enc) / batch_size) / 1000000)))

    encode_avg = float(np.mean(np.array([time[0] for time in times])))
    decode_avg = float(np.mean(np.array([time[1] for time in times])))
    print(f"Average encoding: {encode_avg} ms\nAverage decoding: {decode_avg} ms")