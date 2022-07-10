import torch
import numpy as np

def weighvec_loss(x,h):
    assert x.shape[0] == h.shape[0]
    batch_ww = calculate_weighvec(x)
    latent_ww = calculate_weighvec(h)
    return torch.nn.functional.l1_loss(batch_ww, latent_ww)

def calculate_weighvec(x):
    dist_matrix = torch.cdist(x,x,p=2)
    similarity_matrix = torch.exp(-dist_matrix)
    ww = torch.linalg.solve(similarity_matrix, torch.ones(x.shape[0], 1)).T[0]
    return ww

def calculate_magnitude(x):
    magnitude = torch.sum(calculate_weighvec(x))
    return magnitude
