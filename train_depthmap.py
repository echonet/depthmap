import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision
import numpy as np
import pytorch_lightning as pl
from depth_data import DepthDataset
import wandb
from pathlib import Path


def volume_loss(y_true, y_pred):
    v_true = y_true.sum(dim=(1, 2, 3)) / y_true[0].numel()
    v_pred = y_pred.sum(dim=(1, 2, 3)) / y_pred[0].numel()
    return torch.mean((v_true - v_pred) ** 2)


class AtrousBlock(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=3, dilation=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class ParallelAtrousModel(nn.Module):

    def __init__(self, channel_sizes=[3, 8, 8, 8, 8, 8, 8, 1], dilations=[1, 2, 4, 8, 16]):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([AtrousBlock(in_f, out_f, 3, n) for n in dilations]) for in_f, out_f in zip(channel_sizes[:-1], channel_sizes[1:])])

    def forward(self, x):
        out = None
        for layer in self.layers:
            del out
            out = torch.zeros((x.shape[0], layer[0].out_features, x.shape[2], x.shape[3]), device=x.device)
            for b in layer:
                out += b(x)
            del x
            x = out
        return {'out': out}


class Model(pl.LightningModule):

    def __init__(self, lr=1e-2, volume_weight=0.1, high_res=True):
        super().__init__()
        self.lr = lr
        self.relu = torch.nn.ReLU()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=1)

        if high_res:
            # self.model.classifier = nn.Sequential(
            #         nn.Upsample(scale_factor=2, mode='bilinear'),
            #         self.model.classifier,
            #         nn.Upsample(scale_factor=8, mode='bicubic')
            #     )
            self.model.classifier = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    self.model.classifier,
                    nn.Upsample(scale_factor=8, mode='bicubic')
                )
        
        self.loss_func = torch.nn.MSELoss()
        # self.loss_func = torch.nn.SmoothL1Loss()

        self.epoch_losses = {'train': [], 'val': []}
        self.best_val_loss = 1e10
        self.volume_weight = volume_weight
    
    def forward(self, x):
        o = self.model(x)['out']
        o = self.relu(o) ** 0.5
        return o
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def step(self, batch, step_type='train'):
        x, y_true = batch
        x = torch.movedim(x, -1, 1) / 255
        y_true = y_true[:, None].to(torch.float)
        y_pred = self(x)
        pixel_mse = self.loss_func(y_pred, y_true)
        vol_mse = volume_loss(y_true, y_pred)
        loss = pixel_mse + self.volume_weight * vol_mse

        self.epoch_losses[step_type].append(loss.item())

        log = dict(
            loss=loss.item(),
            pixel_mse=pixel_mse.item(),
            volume_mse=vol_mse.item(),
        )
        log = {k + f'-{step_type}': v for k, v in log.items()}
        wandb.log(log)

        return loss
    
    def training_step(self, batch, i):
        return self.step(batch, 'train')
    
    def validation_step(self, batch, i):
        return self.step(batch, 'val')
    
    def on_epoch_end(self):
        log = {'epoch': self.current_epoch}
        for k in self.epoch_losses:
            log[k + '-epoch-loss'] = np.mean(self.epoch_losses[k])
            self.epoch_losses[k] = []
        wandb.log(log)
        if log['val-epoch-loss'] < self.best_val_loss:
            self.best_val_loss = log['val-epoch-loss']
            torch.save(self.state_dict(), Path(wandb.run.dir) / 'best_model.pt')
            print('Best model so far. Weights saved')


if __name__ == '__main__':

    # import os
    # os.environ['WANDB_MODE'] = 'dryrun'

    config = dict(
        batch_size=200,
        epochs=100,
        lr=1e-2,
        res=(112, 112),
        gpu=1,
        volume_weight=10.0,
        high_res_model=True,
    )

    train_ds = DepthDataset(split='train', res=config['res'])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], num_workers=4)
    val_ds = DepthDataset(split='val', res=config['res'])
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], num_workers=4)
    
    model = Model(lr=config['lr'], high_res=config['high_res_model'], volume_weight=config['volume_weight'])
    trainer = pl.Trainer(gpus=[config['gpu']], max_epochs=config['epochs'])

    wandb.init(project='lv-depthmap', config=config)

    trainer.fit(model, train_dl, val_dl)
