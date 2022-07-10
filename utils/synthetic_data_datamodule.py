import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Sampler, SequentialSampler
from tadasets import swiss_roll, sphere
import snntorch.utils
import pytorch_lightning as pl
import random
from utils import *

class SwissRoll(pl.LightningDataModule):

    def __init__(self, d=3, datapoints=400, r=1, bs=400, multiple=False, rmfrac=0.0, shifts=None):
        super().__init__()
        self.d = d
        self.datapoints = datapoints
        self.r = r
        self.bs = bs
        self.multiple = multiple
        self.rmfrac = rmfrac

    def _generate_sr_data(self, datapoints):
        if self.d==3:
            sr = swiss_roll(n=datapoints, ambient=None, r=3, noise=3/100)
        else:
            sr = swiss_roll(n=datapoints, ambient=self.d, r=3, noise=3/100)
        return sr * self.r

    def setup(self, stage=True):
        self.train_set = torch.as_tensor(self._generate_sr_data(self.datapoints),dtype=torch.float)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, num_workers=8)

class Ball(pl.LightningDataModule):

    def __init__(self, d=3, datapoints=4000, r=5, bs=4000):
        super().__init__()
        self.d = d
        self.datapoints = datapoints
        self.r = r
        self.bs = bs

    def _generate_ball_data(self,datapoints):
        # 3d for now
        ball = np.vstack((np.random.uniform(-self.r, self.r, datapoints * 3),
            np.random.uniform(-self.r, self.r, datapoints * 3),
            np.random.uniform(-self.r, self.r, datapoints * 3))).T
        return ball[np.sum(np.power(ball, 2), 1) < np.power(self.r, 2), ][:datapoints, ]

    def setup(self, stage=True):
        self.train_set = torch.as_tensor(self._generate_ball_data(self.datapoints),dtype=torch.float)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs,shuffle=True,num_workers=8)

class Circle(pl.LightningDataModule):

    def __init__(self, datapoints=4000, r=5, bs=4000):
        super().__init__()
        self.datapoints = datapoints
        self.r = r
        self.bs = bs

    def _generate_circle_data(self,datapoints):
        x = np.random.uniform(-self.r, self.r, 2 * datapoints)
        y = np.random.uniform(-self.r, self.r, 2 * datapoints)
        points = np.vstack((x, y)).T
        return points[np.sqrt(np.sum(points ** 2, 1)) < self.r, ][:datapoints, ]

    def setup(self, stage=True):
        self.train_set = torch.as_tensor(self._generate_circle_data(self.datapoints),dtype=torch.float)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs,shuffle=True,num_workers=8)

class Sphere(pl.LightningDataModule):

    def __init__(self, d=3, datapoints=4000, r=5, bs=4000):
        super().__init__()
        self.d = d
        self.datapoints = datapoints
        self.r = r
        self.bs = bs

    def _generate_sphere_data(self,datapoints):
        return sphere(n=datapoints, r=self.r)

    def setup(self, stage=True):
        self.train_set = torch.as_tensor(self._generate_sphere_data(self.datapoints),dtype=torch.float)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs,shuffle=True,num_workers=8)
