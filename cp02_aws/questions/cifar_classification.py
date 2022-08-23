import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
import argparse
import os
import torchmetrics

class AutoEncoder(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(1024, 10)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, inputs):
        x = F.relu(self.conv1_bn(self.conv1(inputs)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y_true)
        y_pred_label = torch.argmax(y_pred, dim=-1)
        acc = self.train_acc(y_pred_label, y_true)
        self.log("train_loss", loss, prog_bar=False, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y_true)
        y_pred_label = torch.argmax(y_pred, dim=-1)
        acc = self.val_acc(y_pred_label, y_true)
        self.log("val_loss", loss, prog_bar=False, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)

class AWSCallback(pl.callbacks.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.opt.s3_upload_mode == "cp":
            os.system(f"aws s3 cp . {pl_module.opt.aws_s3_uri} --profile {pl_module.opt.aws_profile} --recursive")
        elif pl_module.opt.s3_upload_mode == "sync":
            os.system(f"aws s3 sync . {pl_module.opt.aws_s3_uri} --profile {pl_module.opt.aws_profile} --delete")

class CIFARModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def prepare_data(self):
        self.train_dataset = torchvision.datasets.CIFAR10(
            self.opt.data_dir, train=True, download=True,
            transform=torchvision.transforms.ToTensor())
        self.val_dataset = torchvision.datasets.CIFAR10(
            self.opt.data_dir, train=False, download=True,
            transform=torchvision.transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, num_workers=4, shuffle=False)

def main(opt):
    model = AutoEncoder(opt)
    cifar = CIFARModule(opt)

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=opt.ckpt_dir,
        filename="cifar-{epoch:03d}-{val_acc:.4f}",
        save_top_k=3,
        mode="max"
    )
    aws_cli = AWSCallback()

    if opt.gpus == 0:
        train_flag = {"accelerator":"cpu"}
    elif opt.gpus > 0:
        train_flag = {"accelerator":"gpu", "devices":opt.gpus}

    trainer = pl.Trainer(max_epochs=20, callbacks=[ckpt, aws_cli], **train_flag)
    trainer.fit(model, cifar)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR Training")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--gpus", type=int, default=0) # 0-> CPU, 1 -> use 1 gpu
    parser.add_argument("--s3_upload_mode", type=str, default="sync") # s3のコピーモード。cpかsync
    parser.add_argument("--aws_s3_uri", type=str) # 出力するS3のパスを指定
    parser.add_argument("--aws_profile", type=str) # AWSのプロファイル名を指定

    opt = parser.parse_args()

    os.system(f"rm -rf {opt.ckpt_dir}")
    
    main(opt)