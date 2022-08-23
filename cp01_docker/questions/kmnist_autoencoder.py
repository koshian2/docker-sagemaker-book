import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
import argparse
from PIL import Image
import numpy as np
import os

class AutoEncoder(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.deconv1_bn = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

        self.cache_validation_batch = None

    def forward(self, inputs):
        x = F.relu(self.conv1_bn(self.conv1(inputs)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = torch.sigmoid(self.deconv2(x))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        x_pred = self.forward(x)
        loss = F.l1_loss(x_pred, x)
        self.log("train_loss", loss, prog_bar=False, logger=True)
        return loss

    def write_outputs(self, batch):
        grid = torchvision.utils.make_grid(
            batch[:64], nrow=8, value_range=(0.0, 1.0)
        ) # (C, H, W)
        grid = grid.cpu().numpy().transpose([1, 2, 0]) # (H, W, C)
        grid = (grid * 255.0).astype(np.uint8)

        os.makedirs(self.opt.output_dir, exist_ok=True)
        with Image.fromarray(grid) as img:
            img.save(f"{self.opt.output_dir}/epoch_{self.current_epoch:03}.png")

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        x_pred = self.forward(x)
        if batch_idx == 0:
            # ここで直接吐くとDocker環境で「AssertionError: can only test a child process」という警告文が出る（処理は続く）のでキャッシュさせる
            # ローカル環境ではエラーは出ないので、Docker→ローカルのI/Oボトルネックと、マルチプロセスが噛み合っていないのかも？
            #self.write_outputs(x_pred) 
            self.cache_validation_batch = x_pred
        loss = F.l1_loss(x_pred, x)
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        if self.cache_validation_batch is not None:
            self.write_outputs(self.cache_validation_batch)
            self.cache_validation_batch = None

class KMNISTModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def prepare_data(self):
        self.train_dataset = torchvision.datasets.KMNIST(
            self.opt.data_dir, train=True, download=True,
            transform=torchvision.transforms.ToTensor())
        self.val_dataset = torchvision.datasets.KMNIST(
            self.opt.data_dir, train=False, download=True,
            transform=torchvision.transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=256, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=256, num_workers=4, shuffle=False)

def main(opt):
    model = AutoEncoder(opt)
    mnist = KMNISTModule(opt)

    if opt.gpus == 0:
        train_flag = {"accelerator":"cpu"}
    elif opt.gpus > 0:
        train_flag = {"accelerator":"gpu", "devices":opt.gpus}

    trainer = pl.Trainer(max_epochs=10, **train_flag)
    trainer.fit(model, mnist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Training")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--gpus", type=int, default=0) # 0-> CPU, 1 -> use 1 gpu

    opt = parser.parse_args()

    main(opt)