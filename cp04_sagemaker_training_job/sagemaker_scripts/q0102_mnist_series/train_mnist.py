import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchmetrics
import pytorch_lightning as pl
import argparse
import glob
import os

class MLP(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.view(inputs.shape[0], 784)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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

    def validation_epoch_end(self, outputs):
        print(f"Epoch {self.current_epoch} : Validation acc={self.val_acc.compute():.2%}")
        self.train_acc.reset()
        self.val_acc.reset()

class MNISTModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def prepare_data(self):
        if self.opt.dataset_name == "EMNIST":
            self.train_dataset = torchvision.datasets.EMNIST(
                "./data", split=self.opt.dataset_split, train=True, download=True,
                transform=torchvision.transforms.ToTensor())
            self.val_dataset = torchvision.datasets.EMNIST(
                "./data", split=self.opt.dataset_split, train=False, download=True,
                transform=torchvision.transforms.ToTensor())
        else:
            if self.opt.dataset_name == "MNIST":
                dataset_func = torchvision.datasets.MNIST
            elif self.opt.dataset_name == "FashionMNIST":
                dataset_func = torchvision.datasets.FashionMNIST
            elif self.opt.dataset_name == "KMNIST":
                dataset_func = torchvision.datasets.KMNIST
            else:
                raise NotImplementedError(f"opt.dataset_name {self.opt.dataset_name} is not implemented.")
            self.train_dataset = dataset_func(
                "./data", train=True, download=True,
                transform=torchvision.transforms.ToTensor())
            self.val_dataset = dataset_func(
                "./data", train=False, download=True,
                transform=torchvision.transforms.ToTensor())
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=256, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=256, num_workers=4, shuffle=False)

def get_num_classes(data_module):
    y_max = 0
    for X, y in data_module.train_dataloader():
        y_max = max(y_max, y.max().numpy())
    return y_max + 1

def main(opt):
    # Get Num Classes
    mnist = MNISTModule(opt)
    mnist.prepare_data()
    n_classes = get_num_classes(mnist)

    # Model
    model = MLP(n_classes)

    # Model saving
    exp_name = opt.dataset_name
    if opt.dataset_split != "":
        exp_name += f"-{opt.dataset_split}"

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=f"{opt.ckpt_dir}/{exp_name}",
        filename=exp_name+"-{val_acc:.4f}-{epoch:03d}",
        save_top_k=3,
        mode="max"
    )

    trainer = pl.Trainer(max_epochs=30, callbacks=[ckpt], accelerator="cpu", enable_progress_bar=False)
    trainer.fit(model, mnist)

    # find best checkpoints
    checkpoints = sorted(glob.glob(f"{opt.ckpt_dir}/{exp_name}/{exp_name}*"))
    contents = os.path.basename(checkpoints[-1])
    os.makedirs(opt.result_dir, exist_ok=True)
    with open(f"{opt.result_dir}/{contents}.txt", "w", encoding="utf-8") as fp:
        fp.write(contents)

    print(opt)
    print(contents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Training")
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument("--dataset_split", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="./models")
    parser.add_argument("--result_dir", type=str, default="./results")

    opt = parser.parse_args()

    main(opt)