import os
import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from models import MagAELinear
from utils import *
from synthetic_data_datamodule import SwissRoll, Sphere, Ball
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10,
    "lines.markersize": 3})

###
# Different lambdas (weighvec_loss_penalties)
###

def loss_recons(a, b):
    return torch.nn.functional.mse_loss(a, b, reduction="mean")

def loss_weight(a, b):
    return weighvec_loss(a, b)

torch.manual_seed(42)
np.random.seed(42)

# prepare the dataset
data = SwissRoll(datapoints=1000, bs=1000)
data.setup()

weighvec_loss_pens = (0, 1, 5, 10, 15, 20)

losses_weight = list()
losses_recons = list()

for weighvec_loss_penalty in weighvec_loss_pens:
    min_loss = 10e10
    model = MagAELinear(input_size=3, l=1e-2, weighvec_loss_penalty=weighvec_loss_penalty)
    optimizer = model.configure_optimizers()
    for _ in range(400):
        for i, batch in enumerate(data.train_dataloader()):
            optimizer.zero_grad()
            loss = model.training_step(batch, 0)
            # also get the recon and ww losses separately
            if min_loss > loss.item():
                min_loss = loss.item()
                separate_weighvec_loss = loss_weight(batch, model.encoder(batch).detach()).item()
                separate_recon_loss = loss_recons(batch, model.decoder(model.encoder(batch)).detach()).item()
            loss.backward()
            optimizer.step()
    losses_weight.append(separate_weighvec_loss)
    losses_recons.append(separate_recon_loss)


###
# Different scales
###

scales = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]
min_losses = list()

for s in scales:
    torch.manual_seed(42)
    np.random.seed(42)
    
    data = SwissRoll(datapoints=1000, bs=1000, r=s)
    data.setup()
    model = MagAELinear(input_size=3, l=1e-2, weighvec_loss_penalty=5)
    optimizer = model.configure_optimizers()
    
    # first, MSE loss training
    no_epochs = 400
    min_loss = 10e10
    for e in range(no_epochs): 
       for i, batch in enumerate(data.train_dataloader()):
           optimizer.zero_grad()
           loss = model.training_step(batch, i)
           if min_loss > loss.item():
               min_loss = loss.item()
           loss.backward()
           optimizer.step()
    min_losses.append(min_loss)

## Plots

plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.plot(weighvec_loss_pens, losses_recons, label="Reconstruction")
plt.plot(weighvec_loss_pens, losses_weight, label="Weight")
plt.ylabel("Loss")
plt.xlabel("Lambda")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(scales, min_losses)
plt.ylabel("Loss")
plt.xlabel("Rescale factor")
plt.xscale("log")
plt.yscale("log")
plt.savefig("plot.png", dpi=300)
