#%%
import math
import os
import random
import time
from asyncio import Task

import pytorch_msssim
import tensorboard
from pytorch_msssim import ms_ssim
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
    fig.show()
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
latent_dims = 3*512 # ~= number of float32 values to compress into
deep_n = 2048 # width of hidden deep layers

post_structure_ksize = 17
post_structure_deep_n = 1024
post_structure_latent = 512

ksize = 8 # main convolution kernel size
stride = 2 # main convolution kernel stride
padding = 3 # main convolution kernel application padding
hout = (image_size + 2 * padding - (ksize - 1))/stride
class EncNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder: image -> latent_dims * 4 bytes
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 9, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(8, stride=2),
            nn.Conv2d(8, 4, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4*38*38, deep_n), # output is shape (4,128,128)
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
            nn.ConvTranspose2d(4,8,3,stride=2,padding=1,output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8,8,13,stride=4,padding=0,output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8,8,5,stride=1,padding=0,output_padding=0),
            nn.LeakyReLU()
        )

        self.post_structure_calc = nn.Sequential(
            nn.Conv2d(1,4,15,7),
            nn.LeakyReLU(),
            nn.Conv2d(4,2,7,2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2*15*15, post_structure_deep_n),
            nn.LeakyReLU(),
            nn.Linear(post_structure_deep_n, post_structure_deep_n),
            nn.LeakyReLU(),
            nn.Linear(post_structure_deep_n, post_structure_latent),
            nn.LeakyReLU(),
        )

        self.post_structure_recover = nn.Sequential(
            nn.Linear(post_structure_latent, post_structure_deep_n),
            nn.LeakyReLU(),
            nn.Linear(post_structure_deep_n, post_structure_deep_n),
            nn.LeakyReLU(),
            nn.Linear(post_structure_deep_n, 8 * post_structure_ksize * post_structure_ksize),
            nn.LeakyReLU(),
            nn.Unflatten(1, (8,post_structure_ksize,post_structure_ksize))
        )
    def forward(self, x: Tensor):
        #print(x.shape)
        encoded = self.encoder(x)
        #print(encoded.shape)
        decoded = self.decoder(encoded)
        structure_coded: Tensor = self.post_structure_calc(x) # latent representation is (decoded, structure_coded)


        structure_kernels = self.post_structure_recover(structure_coded)
        batch_dim = structure_kernels.shape[0]
        #print(batch_dim, x.shape, structure_kernels.shape)
        decoded = nn.functional.conv2d(decoded.reshape((-1, 273, 273)), structure_kernels.reshape((batch_dim,-1,) + structure_kernels.shape[2:]), padding=0, groups=batch_dim).unsqueeze(1)
        decoded = decoded[:,:,:256, :256]
        decoded = nn.functional.sigmoid(decoded)
        #print(decoded.shape)
        return decoded.to(torch.float32)

def train(dataloader: DataLoader[HeightTileDataset], mod: nn.Module, loss: nn.Module, opt: torch.optim.Optimizer, writer: SummaryWriter, starti, stop=99999999, write_tensorboard=True, batch_size=1024, subcycles=2):
    loss_nums = []
    for subcycle_i in range(subcycles): # increase number of per-epoch cycles if training is over too quickly before reloading to test
        mod.train()
        batch_losses = []
        for batch, x in enumerate(dataloader):
            input_batch = x
            input_batch = torch.squeeze(input_batch, 1)
            rep_losses = []
            for rep in range(1):
                prediction = mod(input_batch)
                #print(input_batch.shape, "vs", prediction.shape)
                #print(input_batch.shape, prediction.shape)
                assert prediction.shape == input_batch.shape, f"Prediction was shape {prediction.shape}, should be {input_batch.shape}"
                calculated_loss = (loss(prediction.to(torch.float32), input_batch) * mse_loss(prediction, input_batch) * 100.0)

                opt.zero_grad()
                calculated_loss.backward()
                opt.step()
                loss_nums.append(calculated_loss.item())
                rep_losses.append(calculated_loss.item())
                batch_losses.append(calculated_loss.item())
                batch_losses = batch_losses[-max(len(batch_losses), 32):]

                if write_tensorboard:
                    writer.add_scalar("Loss/TrainFine", calculated_loss.item(), (starti + subcycle_i * (len(data_files) / batch_size) + batch) * 1 + rep)
                    writer.flush()

            if batch % 32 == 0:
                print(f"[{int(batch / (len(data_files) / batch_size) * 100):>2d}%]Batch {batch:0>5} ({batch * batch_size} images procd) - loss {float(np.mean(np.array(batch_losses)))}")
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

class SSIM_Loss(pytorch_msssim.MS_SSIM):
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return 100*(1-super(SSIM_Loss, self).forward(X,Y))

if __name__ == '__main__':
    CURR_FIGURE = None
    print("is_main")
    freeze_support()
    torch.multiprocessing.set_start_method('spawn')
    board_writer = SummaryWriter("./runs/")

    # possible batch sizes to test
    batch_options = [2048, 4096, 8192]
    # possible learning rates
    learning_rate_candidates = [1e-6, 1e-5]

    # set the glob expression for the input tif tiles
    tile_folders = ["/run/media/skye/backup31/data/switzerland_tiles/18/*/*.tif", "/run/media/skye/GenericStorage/data/tirol_tiles/part*/18/*/*.tif", "/home/skye/data/swit_tiles2/18/*/*.tif"]
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
    # model = torch.load("models/height13_epX.pt2", weights_only=False).to(device)

    orig_model = model.cpu()

    loss_fn = SSIM_Loss(data_range=1, size_average=True, channel=1, win_size=11)
    mse_loss = torch.nn.MSELoss()
    current_learning_rate = 1e-3 # 1e-5 on fresh model, 1e-4 to 1e-3 for starting trained
    optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate, fused=True) # lr?

    print("Initializing dataloader")
    batch_size = 256 # tweak batch_size to get high standby vram utilisation
    loss_hist = []
    data_loader = torch.utils.data.DataLoader(
        HeightTileDataset(data_files, device),
        batch_size=batch_size, # first dimension of matrices sent,
        shuffle=True, # randomize order
        generator=torch.Generator(device), # load to GPU
        pin_memory=False, # would not work with parallelisation...
        persistent_workers=False) # make workers resident
    print("finding appropriate metaparameters")
    metaparam_data = []
    autodetect_metaparameters = False # set to True to test different batch sizes before training
    autodetect_n_img = 1024*64
    if autodetect_metaparameters:
        losses = {}

        for batch_size_candidate in batch_options:
            print(f"Trying batches with n={batch_size_candidate} ...")


            data_loader = torch.utils.data.DataLoader(
                HeightTileDataset(data_files[:autodetect_n_img], device),
                batch_size=batch_size_candidate,  # first dimension of matrices sent,
                shuffle=True,  # randomize order
                generator=torch.Generator(device),  # load to GPU
                num_workers=4,
                # increase this until CPU utilisation is high or VRAM goes OOM; this is the number of preloading workers
                prefetch_factor=2,  # increase like num_workers, same reasons
                pin_memory=False,  # would not work with parallelisation...
                persistent_workers=True)  # make workers resident
            for learning_rate_candidate in learning_rate_candidates:
                print(f"learning rate {learning_rate_candidate}")
                model = EncNet().to(device)
                model.zero_grad()

                torch.cuda.empty_cache()

                current_learning_rate = learning_rate_candidate  # 1e-5 on fresh model, 1e-4 to 1e-3 for starting trained
                optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate, fused=True)  # lr?
                before_train = time.time()
                train_losses = train(data_loader, model, loss_fn, optimizer, board_writer, 0, autodetect_n_img, False, batch_size_candidate)
                train_duration = time.time() - before_train

                curve_fit_x = np.linspace(0, len(train_losses), len(train_losses))
                curve_fit_y = np.array(train_losses)
                fitted_poly = np.polyfit(curve_fit_x, curve_fit_y, 3)
                curve_points = np.poly1d(fitted_poly)(curve_fit_x)
                delta_real_fit = np.square(curve_points - curve_fit_y)
                f, a = plt.subplots(2,1)
                a[0].plot(curve_fit_y, color="red")
                a[0].plot(curve_points, color="green")
                a[1].plot(delta_real_fit)
                f.show()
                print(float(np.mean(delta_real_fit)))
                mean_start = float(np.mean(np.array(train_losses[:5])))
                mean_end = float(np.mean(np.array(train_losses[-5:])))
                metaparam_data.append((learning_rate_candidate, batch_size_candidate,mean_end - mean_start, train_duration, float(np.mean(delta_real_fit))))
                print(metaparam_data[-1])
                losses[f"lr_{learning_rate_candidate}_bs_{batch_size_candidate}"] = train_losses
                print(f"LR {learning_rate_candidate}, Batch Size {batch_size_candidate}: Improved by {mean_end - mean_start} with variance {float(np.mean(delta_real_fit))} in {train_duration} seconds")
                data = train_losses
                for datum_i in range(len(data)):
                    board_writer.add_scalar(f"Loss/Meta/lr_{learning_rate_candidate}_bs_{batch_size_candidate}", data[datum_i], datum_i)

                    board_writer.flush()
        metaparam_data.sort(key=lambda x: x[2])

        print([f"LR {r[0]}, Batch Size {r[1]}: Improved by {r[2]} with variance {r[4]} in {r[3]} seconds\n" for r in metaparam_data])
        improvement_scaled = [(l[0], l[1], (l[2] / l[3]) * 100 / l[4]) for l in metaparam_data]
        for batch_size_candidate in batch_options:
            print(f"Results for batch size {batch_size_candidate}")
            for learning_rate_candidate in learning_rate_candidates:
                datum = [row for row in improvement_scaled if row[0] == learning_rate_candidate and row[1] == batch_size_candidate][0]
                print(f"[{batch_size_candidate}-batches at LR {learning_rate_candidate}]: score={datum[2]}")
        improvement_scaled.sort(key=lambda x: x[2], reverse=False)

        print(improvement_scaled)
        batch_size = improvement_scaled[0][1]
        current_learning_rate = improvement_scaled[0][0]  # 1e-5 on fresh model, 1e-4 to 1e-3 for starting trained
    else:
        # if not using autodetection, enter learning rate and batch size here
        current_learning_rate = 1e-4
        batch_size = 64 # powers of two are generally useful, for reference: 64 batch_size * 10 num_workers fits easily in 8GB VRAM
        model = orig_model.cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate, fused=True, weight_decay=1e-6)  # lr?
    data_loader = torch.utils.data.DataLoader(
        HeightTileDataset(data_files, device),
        batch_size=batch_size,  # first dimension of matrices sent,
        shuffle=True,  # randomize order
        generator=torch.Generator(device),  # load to GPU
        num_workers=10, # increase this until CPU utilisation is high or VRAM goes OOM; this is the number of preloading workers
        prefetch_factor=1,  # increase like num_workers, same reasons
        pin_memory=False,  # would not work with parallelisation...
        persistent_workers=True,
        drop_last=True
        )  # make workers resident
    for i in range(100):
        print(f"------------------------\nEpoch {i}")
        # each epoch contains train_iter_per_epoch cycles, each train call runs the dataset 4 times

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
            train_losses = train(data_loader, model, loss_fn, optimizer, board_writer, (i * train_iter_per_epoch + j) * 1 * (len(data_files) / batch_size), len(data_files), True, batch_size, num_train_subcycles)
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
        torch.save(model, f"models/height13_ep{i}.pt2")

        try:
            export = torch.export.export(model, (load_tile(fname, device), ))
            torch.export.save(export, f"models/height13_ep{i}_export.pt2")
        except Exception as e:
            print("Error exporting: ", e)

        fig, ax = plt.subplots(1,1)
        ax.plot(loss_hist)
        fig.show()

        if i % 4 == 0:
            current_learning_rate *= 1.3
            optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate, fused=True)  # lr?
