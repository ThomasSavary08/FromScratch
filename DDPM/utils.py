# Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import Resize

# Function(s) definition
def get_cosine_scheduler(T, s = 8e-3):
    def f(t, T):
        return (np.cos((t/T + s)/(1 + s) * (np.pi/2)))**2
    
    alphas = []
    f0 = f(0, T)
    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []
    for t in range(1, T + 1):
        betas.append(min(1. - (alphas[t]/alphas[t-1]), 0.999))
    
    return betas

def extract_coefficients(coeffs: torch.Tensor, indices: torch.Tensor, x_shape: Tuple[int, ...]):
    '''
    Extract coefficients from a torch tensor using another tensor containing the indices to be extracted.

    Args:
        coeffs (torch.Tensor): tensor of dimension (M,) containing the coefficients.
        indices (torch.Tensor): tensor of dimension (N,) containing the indices.
        x_shape (Tuple[int, ...]): dimensions of x to reshape the output.

    Returns:
        res (torch.Tensor): tensor of dimension (N,*((1,)*(len(x.shape) - 1))) containing the coeffients corresponding to the indices.
    '''
    N, *_ = indices.shape
    res = coeffs.gather(-1, indices)
    res = res.reshape(N, *((1,) * (len(x_shape) - 1)))
    return res

def unscale(image: torch.Tensor):
    '''
    Unscale an image from [-1.,1.] to [0.,1.]

    Args:
        image (torch.Tensor): input image to unscale.
        
    Returns:
        res (torch.Tensor): output with the same dimension as the input but with pixels in [0.,1.].
    '''
    res = (image + 1.)/2.
    return res

def plot(image: torch.Tensor):
    '''
    Plot a torch tensor as an image.

    Args:
        image (torch.Tensor): input image of dimension (C,H,W) with pixels in [0.,1.]
    '''
    plt.imshow(image.permute(1,2,0), cmap = 'gray', vmin = 0., vmax = 1.)
    plt.show()

# MNIST dataset
class MNIST(Dataset):
    def __init__(self):
        self.data = load_dataset("mnist", split = 'train', trust_remote_code = True)
        self.data = self.data.with_format("torch")
        self.resize = Resize((32,32), antialias = True)

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, ind):
        img = self.data[ind]['image']
        img = self.resize(img)
        img = 2.*(img/255.) - 1.
        return img