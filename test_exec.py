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
from torch.export import Dim
from torch.fx.experimental.dynamic_spec import ShapeVar, TensorSpec
from torch.utils.data import DataLoader, BatchSampler
from sendable import EncNet
import onnxruntime

import matplotlib.pyplot as plt

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

def save_collage(originals, outputs, n):
    fig, axes = plt.subplots(9,12, figsize=(4000, 3000, "px"), layout="tight")
    for row in range(9):
        for col in range(4):
            ims = [originals[row * 4 + col], outputs[row * 4 + col], np.absolute(originals[row * 4 + col] - outputs[row*4+col])]
            for im_n in range(3):
                ax = axes[row, col * 3 + im_n]

                ax.imshow(ims[im_n].squeeze(0), extent=(0,256,0,256), vmin=0.0, vmax=1.0)
                ax.set_axis_off()
    #plt.show(block=True)
    fig.savefig(n)
    plt.close(fig)

class HeightTileDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, device, transform=None):
        self.images = len(file_list)
        self.image_files=file_list
        self.transform = transform
        self.device = device
    def __len__(self):
        return self.images
    def __getitem__(self, index):
        imdat = load_tile(self.image_files[index])
        if self.transform:
            return self.transform(imdat)
        return imdat

if __name__ == "__main__":
    freeze_support()
    torch.multiprocessing.set_start_method('spawn')
    module = torch.load("models/height25_ep32.pt2", weights_only=False)
    batch = ShapeVar("batch", min=1, max=2048)
    torch.onnx.export(
        model=module.encoder,
        args=(torch.randn(32, 1, 256, 256, dtype=torch.float32).cuda(),),
        f="models/height25_encoder_latest.onnx",
        input_names=["image"],
        output_names=["latent"],
        dynamic_shapes=
            {
                "input": (Dim.DYNAMIC, 1, 256, 256),
            },
        dynamo=True,
        external_data=False,
    )
    torch.onnx.export(
        model=module.decoder,
        args=(torch.randn(32, 2, 12, 12, dtype=torch.float32).cuda(),),
        f="models/height25_decoder_latest.onnx",
        input_names=["latent"],
        output_names=["image"],
        dynamic_shapes={
            "input": (Dim.DYNAMIC, 2,12,12)
        },
        dynamo=True,
        external_data=False,
    )

    onnx_session_enc = onnxruntime.InferenceSession("models/height25_encoder_latest.onnx", providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])
    onnx_session_dec = onnxruntime.InferenceSession("models/height25_decoder_latest.onnx", providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])

    data_files = glob.glob("/home/skye/data/data/switzerland_tiles/18/*/*.tif")
    data_files = list(filter(lambda f: os.path.getsize(f) > 2000,data_files))

    batch_size = 32


    times = []
    for batch_num in range(16):
        random.shuffle(data_files)
        used_files = data_files[:batch_size]
        loaded_files = np.array(list(map(lambda fname: load_tile(fname), used_files))).astype(np.float32)
        print(loaded_files.shape)
        time_start = time.time_ns()
        encoded = onnx_session_enc.run(["latent"], {"image": loaded_files})[0]
        time_after_enc = time.time_ns()
        decoded = onnx_session_dec.run(["image"], {"latent": encoded})[0]
        time_after_dec = time.time_ns()

        num_warmup_batch = 2
        if batch_num < num_warmup_batch:
            print(f"Warmup batch {batch_num}: encoding {(((time_after_enc - time_start) / batch_size) / 1000000):<5f} ms; decoding {(((time_after_dec - time_after_enc) / batch_size) / 1000000):<5f}")
        else:
            print(f"Warmed-up batch {batch_num - num_warmup_batch}: encoding {(((time_after_enc - time_start) / batch_size) / 1000000):<5f} ms; decoding {(((time_after_dec - time_after_enc) / batch_size) / 1000000):<5f}")
            times.append(((((time_after_enc - time_start) / batch_size) / 1000000),(((time_after_dec - time_after_enc) / batch_size) / 1000000)))

    encode_avg = float(np.mean(np.array([time[0] for time in times])))
    decode_avg = float(np.mean(np.array([time[1] for time in times])))
    print(f"Average encoding: {encode_avg} ms\nAverage decoding: {decode_avg} ms")
    print("Showing sample results")
    for imageset in range(16):
        random.shuffle(data_files)
        show_ims = data_files[:9 * 4]
        loaded_show_ims = np.array(list(map(lambda fname: load_tile(fname), show_ims))).astype(np.float32)
        encoded = onnx_session_enc.run(["latent"], {"image": loaded_show_ims})[0]
        encoded = encoded.astype(np.float16).astype(np.float32)
        decoded = onnx_session_dec.run(["image"], {"latent":encoded})[0]
        save_collage(loaded_show_ims, decoded, f"./sampleExec/sample_{imageset}.png")