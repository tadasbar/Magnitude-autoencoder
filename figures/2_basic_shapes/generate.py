import os
import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models import MagAELinear
from utils import *
from synthetic_data_datamodule import SwissRoll, Sphere, Ball
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10,
    "lines.markersize": 3})

torch.manual_seed(42)
np.random.seed(42)

##############
# SWISS ROLL #
##############

# prepare the dataset
data = SwissRoll(datapoints=1000, bs=1000)
data.setup()

model = MagAELinear(input_size=3, l=1e-2, weighvec_loss_penalty=10)
trainer = pl.Trainer(max_epochs=400)
trainer.fit(model, datamodule=data)

input_data_sr = data.train_set.clone()
h_data_sr = model.encoder(data.train_set).detach()

input_weighvec_sr = calculate_weighvec(data.train_set)
h_weighvec_sr = calculate_weighvec(h_data_sr)

corr_sr = np.corrcoef(input_weighvec_sr, h_weighvec_sr)[0, 1]

# PCA
data_pca_sr = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(data.train_set))
weighvec_pca_sr = calculate_weighvec(torch.as_tensor(data_pca_sr, dtype=torch.float32))
corr_pca_sr = torch.corrcoef(torch.vstack((input_weighvec_sr, weighvec_pca_sr)))[0, 1].item()

# TSNE
data_tsne_sr = TSNE(n_components=2).fit_transform(data.train_set)
weighvec_tsne_sr = calculate_weighvec(torch.as_tensor(data_tsne_sr, dtype=torch.float32))
corr_tsne_sr = torch.corrcoef(torch.vstack((input_weighvec_sr, weighvec_tsne_sr)))[0, 1].item()

##########
# SPHERE #
##########

torch.manual_seed(42)
np.random.seed(42)

# prepare the dataset
data = Sphere(datapoints=1000, bs=1000, r=20)
data.setup()

model = MagAELinear(input_size=3, l=1e-2, weighvec_loss_penalty=10)
trainer = pl.Trainer(max_epochs=252)
trainer.fit(model, datamodule=data)

input_data_sphere = data.train_set.clone()
h_data_sphere = model.encoder(data.train_set).detach()

input_weighvec_sphere = calculate_weighvec(data.train_set)
h_weighvec_sphere = calculate_weighvec(h_data_sphere)

corr_sphere = np.corrcoef(input_weighvec_sphere, h_weighvec_sphere)[0, 1]

# PCA
data_pca_sphere = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(data.train_set))
weighvec_pca_sphere = calculate_weighvec(torch.as_tensor(data_pca_sphere, dtype=torch.float32))
corr_pca_sphere = torch.corrcoef(torch.vstack((input_weighvec_sphere, weighvec_pca_sphere)))[0, 1].item()

# TSNE
data_tsne_sphere = TSNE(n_components=2).fit_transform(data.train_set)
weighvec_tsne_sphere = calculate_weighvec(torch.as_tensor(data_tsne_sphere, dtype=torch.float32))
corr_tsne_sphere = torch.corrcoef(torch.vstack((input_weighvec_sphere, weighvec_tsne_sphere)))[0, 1].item()

########
# BALL #
########

torch.manual_seed(42)
np.random.seed(42)

# prepare the dataset
data = Ball(datapoints=1000, bs=1000, r=2)
data.setup()

model = MagAELinear(input_size=3, l=1e-2, weighvec_loss_penalty=10)
trainer = pl.Trainer(max_epochs=194)
trainer.fit(model, datamodule=data)

input_data_ball = data.train_set.clone()
h_data_ball = model.encoder(data.train_set).detach()

input_weighvec_ball = calculate_weighvec(data.train_set)
h_weighvec_ball = calculate_weighvec(h_data_ball)

corr_ball = np.corrcoef(input_weighvec_ball, h_weighvec_ball)[0, 1]

# PCA
data_pca_ball = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(data.train_set))
weighvec_pca_ball = calculate_weighvec(torch.as_tensor(data_pca_ball, dtype=torch.float32))
corr_pca_ball = torch.corrcoef(torch.vstack((input_weighvec_ball, weighvec_pca_ball)))[0, 1].item()

# TSNE
data_tsne_ball = TSNE(n_components=2).fit_transform(data.train_set)
weighvec_tsne_ball = calculate_weighvec(torch.as_tensor(data_tsne_ball, dtype=torch.float32))
corr_tsne_ball = torch.corrcoef(torch.vstack((input_weighvec_ball, weighvec_tsne_ball)))[0, 1].item()

# PLOTTING

fig = plt.figure(figsize=[12, 16])

ax = fig.add_subplot(4, 3, 1, projection='3d')
ax.scatter(*input_data_sr.T, c=input_weighvec_sr, cmap="binary")
ax = fig.add_subplot(4, 3, 2, projection='3d')
ax.scatter(*input_data_sphere.T, c=input_weighvec_sphere, cmap="binary")
ax = fig.add_subplot(4, 3, 3, projection='3d')
ax.scatter(*input_data_ball.T, c=input_weighvec_ball, cmap="binary")

ax = fig.add_subplot(4, 3, 4)
ax.scatter(*h_data_sr.T, c=input_weighvec_sr, cmap="binary")
ax = fig.add_subplot(4, 3, 5)
ax.scatter(*h_data_sphere.T, c=input_weighvec_sphere, cmap="binary")
ax = fig.add_subplot(4, 3, 6)
ax.scatter(*h_data_ball.T, c=input_weighvec_ball, cmap="binary")

# PCA

ax = fig.add_subplot(4, 3, 7)
ax.scatter(*data_pca_sr.T, c=input_weighvec_sr, cmap="binary")
ax = fig.add_subplot(4, 3, 8)
ax.scatter(*data_pca_sphere.T, c=input_weighvec_sphere, cmap="binary")
ax = fig.add_subplot(4, 3, 9)
ax.scatter(*data_pca_ball.T, c=input_weighvec_ball, cmap="binary")

# TSNE

ax = fig.add_subplot(4, 3, 10)
ax.scatter(*data_tsne_sr.T, c=input_weighvec_sr, cmap="binary")
ax = fig.add_subplot(4, 3, 11)
ax.scatter(*data_tsne_sphere.T, c=input_weighvec_sphere, cmap="binary")
ax = fig.add_subplot(4, 3, 12)
ax.scatter(*data_tsne_ball.T, c=input_weighvec_ball, cmap="binary")

plt.tight_layout()
plt.savefig("plot.png", dpi=200)

print("Correlations between the weighting vectors of input and dimensionally reduced datasets")
print(f"corr_sr: {corr_sr}")
print(f"corr_pca_sr: {corr_pca_sr}")
print(f"corr_tsne_sr: {corr_tsne_sr}")
print("")
print(f"corr_sphere: {corr_sphere}")
print(f"corr_pca_sphere: {corr_pca_sphere}")
print(f"corr_tsne_sphere: {corr_tsne_sphere}")
print("")
print(f"corr_ball: {corr_ball}")
print(f"corr_pca_ball: {corr_pca_ball}")
print(f"corr_tsne_ball: {corr_tsne_ball}")
