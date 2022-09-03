import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import timm
import torchmetrics
from timm.scheduler import CosineLRScheduler
from PIL import Image, ImageOps
import argparse
import os
import glob

class EfficientNetB0(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("efficientnet_b0", num_classes=5, pretrained=True)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        scheduler = CosineLRScheduler(optimizer, t_initial=50, 
            lr_min=1e-7, warmup_t=5, warmup_lr_init=1e-7, warmup_prefix=True)
        return [optimizer, ], [scheduler, ]

    def lr_scheduler_step(self, scheduler, optmizer_idx, metric):
        scheduler.step(self.current_epoch+1)

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

class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform):
        super().__init__()
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index][0]) as fp:
            if self.transform is not None:
                image = self.transform(fp)
            else:
                image = image        
        return image, self.files[index][1]

class FlowersModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        dirs = sorted(x for x in glob.glob(opt.data_dir) if os.path.isdir(x))
        class_keys = {d:i for i, d in enumerate(dirs)}
        print(class_keys)

        all_data = torchvision.datasets.DatasetFolder.make_dataset(
            opt.data_dir,  class_keys, is_valid_file=self.is_valid_file)
        print("Valid images =", len(all_data))
        self.train_data, self.val_data = train_test_split(
            all_data, test_size=0.2, random_state=123, shuffle=True)

    def is_valid_file(self, file_path):
        try:
            with Image.open(file_path) as img:
                if img.mode not in ["RGB", "RGBA"]:
                    return False
                ImageOps.invert(img)
        except:
            return False
        return True

    def prepare_data(self):
        self.trainset = FlowersDataset(
            self.train_data, transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()       
        ]))
        self.valset = FlowersDataset(
            self.val_data, transform=transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(1, 1), ratio=(1, 1)), # determistic crop
                transforms.ToTensor()
        ]))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset, batch_size=64, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset, batch_size=64, shuffle=False, num_workers=4)

def main(opt):
    model = EfficientNetB0()
    dataset = FlowersModule(opt)

    # Model saving
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=opt.ckpt_dir,
        filename="mnist-{epoch:03d}-{val_acc:.4f}",
        save_top_k=3,
        mode="max"
    )

    trainer = pl.Trainer(max_epochs=30, callbacks=[ckpt], accelerator="gpu", devices=1, enable_progress_bar=False)
    trainer.fit(model, dataset) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF Flowers")
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/training/flower_photos")
    parser.add_argument("--ckpt_dir", type=str, default="/opt/ml/checkpoints")

    opt = parser.parse_args()
    main(opt)