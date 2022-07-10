import os
import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from models import MagAEConv
from utils import *
from synthetic_data_datamodule import SwissRoll, Sphere, Ball
from torchvision import datasets, transforms
import snntorch.utils
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams.update({"font.size": 10,
    "lines.markersize": 3})

# idea: take e.g. 32 elements, which will be a part of 64 set; M - 64x64 mat, A - 32x32 mat
# then take other 32 elements, then this M-1 iwll be athe A in the prev iteration

torch.manual_seed(42)
np.random.seed(42)

ds_divisor = 20
ds_size = 60000 // ds_divisor
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,
    transform=transforms.ToTensor())
snntorch.utils.data_subset(mnist_trainset, ds_divisor)

fixed_data = torch.vstack(tuple(map(lambda i: mnist_trainset[i][0].detach(),
    range(len(mnist_trainset)))))

###
# fFind when the total loss is minimized
###

losses_total = list()
losses_ww = list()
losses_recon = list()

torch.manual_seed(42)
np.random.seed(42)

model = MagAEConv(rescale_factor=0.6, l=1e-3, wwloss_penalty=5)
optimizer = model.configure_optimizers()
no_mse_epochs = 5
no_ww_epochs = 100
for em in range(no_mse_epochs):
    print(f"em: {em}")
    for i, batch in enumerate(torch.utils.data.DataLoader(mnist_trainset, batch_size=100)):
        # MSE opt
        optimizer.zero_grad()
        loss = model.training_step_mse(batch, em)
        loss.backward()
        optimizer.step()

for ew in range(no_ww_epochs):
    print(f"ew: {ew}")
    for i, batch in enumerate(torch.utils.data.DataLoader(mnist_trainset, batch_size=ds_size)):
        optimizer.zero_grad()
        loss, loss_ww, loss_recon = model.training_step_both(batch, ew)
        loss.backward()
        optimizer.step()
        losses_total.append(loss.item())
        losses_ww.append(loss_ww.item())
        losses_recon.append(loss_recon.item())

###
# Get the model which minimizes total loss
###

torch.manual_seed(42)
np.random.seed(42)

model = MagAEConv(rescale_factor=0.6, l=1e-3, wwloss_penalty=5)
optimizer = model.configure_optimizers()
no_mse_epochs = 5
no_ww_epochs = np.argmin(losses_total) + 1
for em in range(no_mse_epochs):
    print(f"em: {em}")
    for i, batch in enumerate(torch.utils.data.DataLoader(mnist_trainset, batch_size=100)):
        # MSE opt
        optimizer.zero_grad()
        loss = model.training_step_mse(batch, em)
        loss.backward()
        optimizer.step()

for ew in range(no_ww_epochs):
    print(f"ew: {ew}")
    for i, batch in enumerate(torch.utils.data.DataLoader(mnist_trainset, batch_size=ds_size)):
        optimizer.zero_grad()
        loss, _, _ = model.training_step_both(batch, ew)
        loss.backward()
        optimizer.step()



data_h = model.encoder(fixed_data.reshape(ds_size, 1, 28, 28)).detach()
ww_h = calculate_weighvec(data_h * 0.6)
ww_input = calculate_weighvec(fixed_data.reshape(ds_size, 28 * 28) * 0.6)

correlation_ww = torch.corrcoef(torch.vstack((ww_input, ww_h)))[0, 1].item()

# PCA
data_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(fixed_data.reshape(ds_size, 28*28)))
ww_pca = calculate_weighvec(torch.as_tensor(data_pca, dtype=torch.float32))
corr_pca = torch.corrcoef(torch.vstack((ww_input, ww_pca)))[0, 1].item()

# TSNE
data_tsne = TSNE(n_components=2).fit_transform(fixed_data.reshape(ds_size, 28*28))
ww_tsne = calculate_weighvec(torch.as_tensor(data_tsne, dtype=torch.float32))
corr_tsne = torch.corrcoef(torch.vstack((ww_input, ww_tsne)))[0, 1].item()

epochs = list(range(100))

plt.figure(figsize=(10,10))

plt.subplot(2, 2, 1)
plt.plot(epochs, losses_ww, label="Weight")
plt.plot(epochs, losses_recon, label="Reconstruction")
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(*data_h.T, c=mnist_trainset.targets)
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(2, 2, 3)
plt.scatter(*data_pca.T, c=mnist_trainset.targets)
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(2, 2, 4)
plt.scatter(*data_tsne.T, c=mnist_trainset.targets)
plt.xlabel("x")
plt.ylabel("y")

plt.savefig("plot.png", dpi=300)


print(f"Correlation between input and 2D weighting vectors: {correlation_ww}")
print(f"corr_pca: {corr_pca}")
print(f"corr_tsne: {corr_tsne}")
