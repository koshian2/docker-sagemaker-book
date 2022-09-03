import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchmetrics
import pytorch_lightning as pl
import argparse
 
class ConvNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(3):
            for j in range(3):
                if i == 0 and j == 0:
                    in_chs = 3
                elif j==0:
                    in_chs = 64 * (2**(i-1))
                else:
                    in_chs = 64 * (2**i)
                out_chs = 64 * (2**i)
                self.layers.append(nn.Conv2d(in_chs, out_chs, 3, padding=1))
                self.layers.append(nn.BatchNorm2d(out_chs))
                
                if i != 2 and j != 2:
                    self.layers.append(nn.ReLU(inplace=True))
                if i != 2 and j == 2:
                    self.layers.append(nn.AvgPool2d(2))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, inputs):
        x = inputs
        for l in self.layers:
            x = l(x)
        x = self.global_pool(x).view(x.shape[0], 256)
        x = self.fc(x)
        return x
 
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], gamma=0.2)
        return [optimizer,], [scheduler,]
 
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
        print(f"\nEpoch  {self.current_epoch:03}  | ValidationAcc={self.val_acc.compute():.2%}")
        self.val_acc.reset()
 
class CIFARModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
 
    def prepare_data(self):
        self.train_dataset = torchvision.datasets.CIFAR10(
            self.opt.data_dir, train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop((32, 32), padding=2),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()
            ]))
        self.val_dataset = torchvision.datasets.CIFAR10(
            self.opt.data_dir, train=False, download=True,
            transform=torchvision.transforms.ToTensor())
 
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128, num_workers=4, shuffle=True)
 
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, num_workers=4, shuffle=False)
 
def main(opt):
    model = ConvNet()
    data = CIFARModule(opt)
    logger = pl.loggers.TensorBoardLogger(f"{opt.ckpt_dir}/logs")
 
    # Model saving
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=opt.ckpt_dir,
        filename="cifar-10-{epoch:03d}-{val_acc:.4f}",
        save_top_k=3,
        mode="max"
    )
 
    trainer = pl.Trainer(
        max_epochs=100, callbacks=[ckpt],
        accelerator="gpu", devices=1, logger=logger)
    trainer.fit(model, data)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Training")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
 
    opt = parser.parse_args()
 
    main(opt)