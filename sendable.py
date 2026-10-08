#%%
import math
import os
import random
import time
from asyncio import Task
from typing import Optional, List, Union, Tuple

import pytorch_msssim
import tensorboard
from pytorch_msssim import ms_ssim
from torch.amp import custom_fwd
from torch.multiprocessing import freeze_support
import numpy as np


import torch
import glob

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from torch import nn, Tensor
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter


CURR_FIGURE=None

def load_tile(fname, device):
    im = (np.array(Image.open(fname).get_flattened_data()).reshape(1, image_size, image_size))
    min_val = np.amin(im)
    max_val = np.amax(im)
    #print(max_val - min_val)
    return torch.from_numpy((im - min_val) / (max(max_val - min_val, 1))).unsqueeze(0).to(device, torch.float)

unloader = transforms.ToPILImage()  # reconvert into PIL image


def get_im(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    return unloader(image)
def imshow(tensorA, tensorB, title=None):
    global CURR_FIGURE

    if CURR_FIGURE:
        try: plt.close(CURR_FIGURE)
        except: print("e")
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(get_im(tensorA))
    ax[1].imshow(get_im(tensorB))

    if title is not None:
        ax[0].title(title)
    #fig.show()
    CURR_FIGURE = fig


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


image_size = 256 # image side length
latent_dims = 512 # ~= number of float32 values to compress into
deep_n = 1024 # width of hidden deep layers

class EncNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder: image -> latent_dims * 4 bytes
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 9, stride=3, padding=0),
            nn.MaxPool2d(8, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 4, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
            nn.Flatten(),
            nn.Linear(4*38*38, deep_n),
            nn.LeakyReLU(),
            nn.Linear(deep_n, deep_n),
            nn.LeakyReLU(),
            nn.Linear(deep_n, latent_dims),
            nn.LeakyReLU()
        )
        # decoder is pretty directly mirror, but trained independently
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, deep_n),
            nn.LeakyReLU(),
            nn.Linear(deep_n, deep_n),
            nn.LeakyReLU(),
            nn.Linear(deep_n, 4*33*33),
            nn.LeakyReLU(),
            nn.Unflatten(1, (4,33,33)),
            nn.Conv2d(4,8,3,padding="same"),
            nn.LeakyReLU(),
            nn.Upsample((266,266)),
            nn.LeakyReLU(),
            nn.Conv2d(8,4,11, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(4,1,5,padding=0),
            nn.Sigmoid()
        )
    def forward(self, x: Tensor):
        #print(x.shape)
        encoded = self.encoder(x)
        #print(encoded.shape)
        decoded = self.decoder(encoded)

        #print(decoded.shape)
        return decoded
def save_collage(originals, outputs, n_step, vanity_folder):
    fig, axes = plt.subplots(9,12, figsize=(1200, 900, "px"), layout="tight")
    for row in range(9):
        for col in range(6):
            ims = [originals[row * 6 + col], outputs[row * 6 + col]]
            for im_n in range(2):
                ax = axes[row, col * 2 + im_n]

                ax.imshow(ims[im_n], extent=(0,256,0,256))
                ax.set_axis_off()
    fig.savefig(f"{vanity_folder}/step{n_step:08d}.png")

    plt.close(fig)

def train(dataloader: DataLoader[HeightTileDataset], mod: nn.Module, loss: nn.Module, opt: torch.optim.Optimizer, writer: SummaryWriter, starti, stop=99999999, write_tensorboard=True, batch_size=1024, subcycles=2, scaler=torch.amp.GradScaler("cuda"), epoch_num = 0, animation = False):
    loss_nums = []
    for subcycle_i in range(subcycles): # increase number of per-epoch cycles if training is over too quickly before reloading to test
        mod.train()
        batch_losses = []
        test_ims = []
        for batch, x in enumerate(dataloader):
            input_batch = x
            if batch == 0:
                test_ims = x[:54,:,:,:]
            input_batch = torch.squeeze(input_batch, 1)
            rep_losses = []
            for rep in range(1):
                #with torch.autocast("cuda", torch.float32):

                prediction = mod(input_batch)
                #print(input_batch.shape, "vs", prediction.shape)
                #print(input_batch.shape, prediction.shape)
                assert prediction.shape == input_batch.shape, f"Prediction was shape {prediction.shape}, should be {input_batch.shape}"
                loss1 = loss(prediction, input_batch)
                calculated_loss = loss1.item()
                scaler.scale(loss1).backward()
                scaler.step(opt)
                scaler.update()
                #opt.zero_grad()

                loss_nums.append(calculated_loss)
                rep_losses.append(calculated_loss)
                batch_losses.append(calculated_loss)
                batch_losses = batch_losses[-min(len(batch_losses), 32):]

                if write_tensorboard:
                    writer.add_scalar("Loss/TrainFine", calculated_loss, (starti + subcycle_i * (len(data_files) / batch_size) + batch) * 1 + rep)
                    writer.flush()

            if batch % 32 == 0:
                print(f"[{int(batch / (len(data_files) / batch_size) * 100):>2d}%]Batch {batch:0>5} ({batch * batch_size} images procd) - loss {float(np.mean(np.array(batch_losses)))}")
            if batch % 16 == 0 and animation:
                im_batch = model(test_ims.squeeze(1).cuda())
                save_collage([get_im(test_ims[im].squeeze(0)) for im in range(len(test_ims))], [get_im(im_batch[im].squeeze(0)) for im in range(len(test_ims))], batch // 4, f"vanity{epoch_num:02}")
            if batch % math.floor(10000/batch_size) == 0:
                disp_im = load_tile(data_files[random.randrange(0, len(data_files))], device)
                imshow(disp_im.squeeze(0), model(disp_im).squeeze(0))
                if write_tensorboard:
                    fig, axes = plt.subplots(1,2)
                    axes[0].imshow(get_im(disp_im.squeeze(0)))
                    axes[1].imshow(get_im(model(disp_im)[0].to(device, torch.float32)))
                    writer.add_figure("PartialResult", fig, int(batch / math.floor(10000/batch_size)))
                
            if batch * batch_size > stop:
                break
    print("Training cycle done")
    return loss_nums



def test(dataloader: DataLoader, mod: nn.Module, loss: nn.Module):
    num_batches = 40 # number of batches to test over (shuffled)
    mod.eval()
    test_loss, n_correct = 0,0
    with torch.no_grad():
        count = 0
        for X in dataloader:
            count += 1
            if count >= num_batches:
                break

            X = X.to(device)
            X = X.squeeze(1)
            pred = mod(X)

            test_loss += loss(pred,X).item()
            if count % 10 == 0:
                print(f"test batch {count}: {loss(pred, X).item()}")
    test_loss /= num_batches
    #print(f"Avg loss: {test_loss:>8f}")
    return test_loss

class DeviationCorrectedLoss(pytorch_msssim.SSIM):
    def __init__(
        self,
        data_range: float = 1.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    ):
        super(DeviationCorrectedLoss, self).__init__(data_range, True, win_size, win_sigma, channel, spatial_dims, K)
        self.mse_loss = nn.MSELoss()


    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        X_deviation = X.std(3)
        X_mean = X.mean(dim=3)
        Y_deviation = Y.std(3)
        Y_mean = Y.mean(dim=3)
        delta_mean = ((Y_mean - X_mean) * 4) ** 2
        #print(delta_mean.mean(), (((Y_mean - X_mean) * 5) ** 2).mean())
        delta_deviation = ((Y_deviation - X_deviation) * 10) ** 2
        return 0.25 * (((1 - super(DeviationCorrectedLoss, self).forward(X, Y)) * 200) + (self.mse_loss(X,Y) * 200) + delta_deviation.mean() + delta_mean.mean())

if __name__ == '__main__':
    CURR_FIGURE = None
    print("is_main")
    freeze_support()

    scaler = torch.amp.GradScaler("cuda")
    torch.multiprocessing.set_start_method('spawn')
    board_writer = SummaryWriter("./runs/")

    # set the glob expression for the input tif tiles
    tile_folders = ["/run/media/skye/backup31/dat/switzerland_tiles/18/**/*.tif"]
    zoom_level = "**"
    data_files = []
    for glob_expr in tile_folders:
        data_files += glob.glob(glob_expr)
    print(f"{len(data_files)} raw files")

    # exclude files under 1.2k (empty images from outside the mapped area)
    data_files = list(filter(lambda x: os.path.getsize(x) > 2000, data_files))
    if not len(data_files):
        print("Error: No tiles!")
        exit(2)
    print(f"{len(data_files)} tiles ready")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    torch.cuda.empty_cache()
    model = EncNet().to(device)

    # uncomment and adjust path to load checkpointed model
    #model = torch.load("models/height18_ep0.pt2", weights_only=False).to(device)


    loss_fn = DeviationCorrectedLoss(data_range=1, channel=1, win_size=3)
    current_learning_rate = 1e-4
    batch_size = 128 # powers of two are generally useful, for reference: 64 batch_size * 10 num_workers fits easily in 8GB VRAM
    loss_hist = []

    # whether to create 16*6 mosaics every 16 batches (same images, usable as animation)
    make_animation_frames = True

    optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate, fused=True, weight_decay=1e-5)  # lr?
    data_loader = torch.utils.data.DataLoader(
        HeightTileDataset(data_files, device),
        batch_size=batch_size,  # first dimension of matrices sent,
        shuffle=True,  # randomize order
        generator=torch.Generator(device),  # load to GPU
        num_workers=8, # increase this until CPU utilisation is high or VRAM goes OOM; this is the number of preloading workers
        prefetch_factor=1,  # increase like num_workers, same reasons
        pin_memory=False,  # would not work with parallelisation...
        persistent_workers=True,
        drop_last=True
        )  # make workers resident
    for i in range(100):
        print(f"------------------------\nEpoch {i}")
        # each epoch contains train_iter_per_epoch cycles, each train call runs the dataset 4 times
        os.mkdir(f"vanity{i:02}")
        # display current inference result
        fig, axes = plt.subplots(1, 2)

        fname = data_files[random.randrange(0, len(data_files))]
        imshow(load_tile(fname, device)[0], model(load_tile(fname, device))[0])
        board_writer.add_figure("CurrentResult", figure=fig, global_step=i * 3 )

        # run training/test loop n times
        train_iter_per_epoch = 1 if i < 2 else 1
        for j in range(train_iter_per_epoch):
            print(f"-------------\n[[{int(j/train_iter_per_epoch * 100.0):>2d}%]]\n-----------")

            num_train_subcycles = 1 if i < 1 else 1
            tick = time.time()
            train_losses = train(data_loader, model, loss_fn, optimizer, board_writer, (i * train_iter_per_epoch + j) * 1 * (len(data_files) / batch_size), len(data_files), True, batch_size, num_train_subcycles,scaler=scaler, epoch_num=i, animation=make_animation_frames)
            time_full_train = (time.time() - tick) / (len(data_files) * num_train_subcycles) # FIXME pull this scalar from the same place as train() subepoch count
            print(f"average {(time_full_train * 1000):2<5f}ms training time per image (approx)")
            loss_hist = loss_hist + train_losses
            test_losses = test(data_loader, model, loss_fn)
            board_writer.add_scalar("Loss/test", test_losses, i * train_iter_per_epoch + j + 1 )
            board_writer.flush()

            loss_hist.append(test_losses)

            board_writer.add_scalar("LearningRate", current_learning_rate, global_step=i * 3 + j + 1)
            for k in range(10):
                fig, axes = plt.subplots(1, 2)
                fname = data_files[random.randrange(0,len(data_files))]
                if CURR_FIGURE:
                    try: plt.close(CURR_FIGURE)
                    except: print("e")
                axes[0].imshow(get_im(load_tile(fname, device)[0]))
                axes[1].imshow(get_im(model(load_tile(fname, device))[0]))
                board_writer.add_figure("CurrentResult", figure= fig,global_step=(i * 1 + j) * 10 + k)
            CURR_FIGURE = fig
            board_writer.flush()
        # save full model (encode+decode, need class definition to load, but weights are saved)
        # should be 24.0MiB
        torch.save(model, f"models/height18_ep{i}.pt2")

        try:
            export = torch.export.export(model, (load_tile(fname, device), ))
            torch.export.save(export, f"models/height18_ep{i}_export.pt2")
        except Exception as e:
            print("Error exporting: ", e)

        fig, ax = plt.subplots(1,1)
        ax.plot(loss_hist)
        #fig.show()

        if i % 4 == 0:
            current_learning_rate *= 1.3
            optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate, fused=True)  # lr?
