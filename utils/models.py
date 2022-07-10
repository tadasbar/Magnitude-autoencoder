import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import *

class MagAELinear(pl.LightningModule):
    
    def __init__(self, input_size, weighvec_loss_penalty=20, l=1e-2):
    
        super(MagAELinear,self).__init__()
        self.input_size = input_size
        self.l = l
        self.weighvec_loss_penalty = weighvec_loss_penalty
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, self.input_size)
        )
    
    def training_step(self,batch,batch_idx):
        x = batch
        h = self.encoder(x)
        x_hat = self.decoder(h)
        weve_loss = weighvec_loss(x, h) * self.weighvec_loss_penalty
        re_loss = torch.nn.functional.mse_loss(x, x_hat, reduction="mean")
        return weve_loss + re_loss

    def training_step_mse(self, batch):
        x = batch
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return torch.nn.functional.mse_loss(x, x_hat, reduction="mean")

    def training_step_mse_only(self, h, x):
        x_hat = self.decoder(h)
        return torch.nn.functional.mse_loss(x, x_hat, reduction="mean")

    def training_step_mse(self, batch):
        x = batch
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return torch.nn.functional.mse_loss(x, x_hat, reduction="mean")
    
    def validation_step(self,batch,batch_idx):
        x = batch
        h = self.encoder(x)
        loss = weighvec_loss(x, h)
        
        self.log('reconstruction_loss', loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.l,weight_decay=1e-5)
        return optimizer

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)

class MagAEConv(pl.LightningModule):
    """Convolutional Autoencoder with 2d latent space.
    Based on the architecture from:
        Model 1 in 
        `A Deep Convolutional Auto-Encoder with Pooling - Unpooling Layers in
        Caffe - Volodymyr Turchenko, Eric Chalmers, Artur Luczak`
    """
    
    def __init__(self, rescale_factor, weighvec_loss_penalty=20, l=1e-3, input_channels=1):
    
        super(MagAEConv,self).__init__()
        self.l = l
        self.weighvec_loss_penalty = weighvec_loss_penalty
        self.rescale_factor = rescale_factor
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 8, 9, stride=1, padding=0),  # b, 16, 10, 10
            nn.SiLU(),
            nn.Conv2d(8, 4, 9, stride=1, padding=0),  # b, 2, 3, 3
            nn.SiLU(),
            View((-1, 576)),

            nn.Linear(576, 1000),
            nn.SiLU(True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 500),
            nn.SiLU(True),
            nn.BatchNorm1d(500),
            nn.Linear(500, 250),
            nn.SiLU(True),
            nn.BatchNorm1d(250),
            nn.Linear(250, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 250),
            nn.SiLU(True),
            nn.BatchNorm1d(250),
            nn.Linear(250, 500),
            nn.SiLU(True),
            nn.BatchNorm1d(500),
            nn.Linear(500, 1000),
            nn.SiLU(True),

            View((-1, 1000, 1, 1)),
            nn.SiLU(),
            nn.ConvTranspose2d(1000, 4, 12, stride=1, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(4, 4, 17, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(4, 1, 1, stride=1, padding=0),
            nn.Tanh()
        )

    def training_step_mse(self,batch,batch_idx):
        x, y = batch
        h = self.encoder(x)
        x_hat = self.decoder(h)
        re_loss = torch.nn.functional.mse_loss(x, x_hat, reduction="mean")
        #print(f"epoch no: {batch_idx}; re_loss: {re_loss}")
        return re_loss
    
    def training_step_ww(self,batch,batch_idx):
        x, y = batch
        h = self.encoder(x)
        weve_loss = weighvec_loss(x.reshape(x.shape[0], 1 * 28 * 28) * self.rescale_factor,
            h * self.rescale_factor) * self.weighvec_loss_penalty
        return weve_loss

    def training_step_both(self,batch,batch_idx):
        x, y = batch
        h = self.encoder(x)
        x_hat = self.decoder(h)
        weve_loss = weighvec_loss(x.reshape(x.shape[0], 1 * 28 * 28) * self.rescale_factor,
            h * self.rescale_factor) * self.weighvec_loss_penalty
        re_loss = torch.nn.functional.mse_loss(x, x_hat, reduction="mean")
        print(f"weve_loss: {weve_loss.item()}; recon_loss: {re_loss.item()}")
        #print(f"epoch no: {batch_idx}; re_loss: {re_loss}; weve_loss: {weve_loss}")
        return re_loss + weve_loss, weve_loss, re_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.l,weight_decay=1e-5)
        return optimizer

