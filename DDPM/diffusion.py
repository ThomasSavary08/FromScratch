# Libraries
import tqdm
import utils
import torch
from tqdm import tqdm
from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

class DiffusionModel():
    '''
    Diffusion model from DDPM article

    Args:
        model (torch.nn.Module): neural network architecture (UNet, ViT) to predict epsilon_theta.
        input_shape (Tuple[int, int , int]): dimension of the input, should be (C_in, H_in, W_in).
        device (str): device on which computations are performed.
        T (int): maximum timestep for diffusion process.
        scheduler (str): noise schedule to use for the difussion process.
    '''
    def __init__(self, model: torch.nn.Module, input_shape: Tuple[int, int, int], device: str, T: int = 4000, scheduler: str = "cosine"):
        # Instanciate the model to predict episilon_theta
        self.device = device
        self.model = model.to(device)
        self.C, self.H, self.W = input_shape

        # Instanciate the coefficients of diffusion
        self.T = T
        if (scheduler == "linear"):
            self.betas = torch.linspace(1e-4, 2e-2, T, dtype = torch.float32, requires_grad = False, device = self.device)
        else:
            self.betas = torch.tensor(utils.get_cosine_scheduler(T), dtype = torch.float32, requires_grad = False, device = self.device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim = -1, dtype = torch.float32)
    
    @torch.no_grad()
    def diffusion_process(self, x_batch: torch.Tensor, n_steps: int):
        '''
        Apply the forward process to a batch of images drawn from q(x_0).

        Args:
            x_batch (torch.Tensor): tensor of dimension (N,self.C,self.H,self.W) containing a batch of input images.
        
        Returns:
            res (torch.Tensor): tensor of dimension (n_steps+1,N,self.C,self.H,self.W) contaning the input tensor at each time step of the forward process.
        '''
        # Create a tensor to store the evolution of the process.
        assert (n_steps <= self.T)
        res = torch.zeros(n_steps + 1, *x_batch.shape, dtype = torch.float32, device = self.device)
        res[0,:] = x_batch

        # Diffusion process
        for t in range(1, n_steps+1):
            c1, c2 = torch.sqrt(self.alphas[t-1]), torch.sqrt(self.betas[t-1])
            noise = torch.randn_like(x_batch)
            res[t,:] = c1*res[t-1,:] + c2*noise

        return res
    
    @torch.no_grad()
    def draw_time(self, batch_size: int):
        '''
        Draw times from an uniform distribution

        Args:
            batch_size (int): number of samples in a batch.

        Return:
            times (torch.Tensor): tensor with dimension (batch_size,) containing times.
        '''
        times = torch.randint(1, self.T+1, size = (batch_size,), dtype = torch.long, device = self.device)
        return times
    
    @torch.no_grad()
    def sampling(self, n_samples: int):
        '''
        Sampling of n_samples using the learned reverse process.

        Args:
            n_samples (int): number of samples to generate.

        Returns:
            res (torch.Tensor): tensor with dimension (self.T+1,n_samples,self.C,self.H,self.W) containing the reverse process of n_samples.
        '''
        # Draw samples from a normal distribution N(0,1)
        samples = torch.randn(n_samples, self.C, self.H, self.W, dtype = torch.float32, device = self.device)
        res = torch.zeros(self.T+1, *samples.shape, dtype = torch.float32, device = self.device)
        res[-1,:] = samples
        for t in range(self.T-1, -1, -1):
            if (t+1) > 1:
                z = torch.randn_like(samples, dtype = torch.float32, device = self.device)
            else:
                z = torch.zeros_like(samples, dtype = torch.float32, device = self.device)
            eps_theta = self.model(samples, torch.tensor([t+1], dtype = torch.long, device = self.device).repeat(n_samples))
            c1 = 1./torch.sqrt(self.alphas[t])
            c2 = (1. - self.alphas[t])/torch.sqrt(1. - self.alphas_bar[t])
            c3 = torch.sqrt(self.betas[t])
            samples = c1*(samples - c2*eps_theta['sample']) + c3*z
            res[t,:] = samples
        return res

    
    @torch.no_grad()
    def draw_noise(self, x: torch.Tensor):
        '''
        Draw noise from a normal distribution N(0,1).

        Args:
            x (torch.Tensor): tensor on which noise will be added (in order to copy its dimensions).
        
        Returns:
            noise (torch.Tensor): tensor with the same dimension as x and drawn from a normal distribution N(0,1).
        '''
        noise = torch.randn_like(x, dtype = torch.float32, device = self.device)
        return noise
    
    @torch.no_grad()
    def draw_xt(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        '''
        Draw x_t knowing x_0 from q(x_t|x_0) ~ N(sqrt(alphabar_t)x_0, (1-alphabar_t)I).

        Args:
            x_0 (torch.Tensor): tensor with dimension (N,C,H,W) drawn from q(x_0).
            noise (torch.Tensor): tensor with dimension (N,C,H,W) drawn from N(0,I).
            t (torch.Tensor): tensor with dimension (N) drawn from a uniform distribution.
        
        Returns:
            x_t (torch.Tensor): tensor with the same dimension as x_0 drawn from q(x_t|x_0).
        '''
        c1 = torch.sqrt(utils.extract_coefficients(self.alphas_bar, t - 1, x_0.shape))
        c2 = torch.sqrt(1. - utils.extract_coefficients(self.alphas_bar, t - 1, x_0.shape))
        x_t = c1*x_0 + c2*noise
        x_t.requires_grad = True
        assert (x_t.device.type == self.device)
        return x_t

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor):
        '''
        Predict the noise epsilon knowing x_t and t.

        Args:
            x_t (torch.Tensor): tensor with dimension (N,C,H,W).
            t (torch.Tensor): tensor of dimension (N) containing times.
        
        Returns:
            eps_theta (torch.Tensor): tensor with the same dimension as x_t containg an estimation of the noise.
        '''
        # Check dimensions
        N, C, H, W = x_t.shape
        assert (C == self.C)
        assert (H == self.H)
        assert (W == self.W)

        # Check devices
        xt_device = x_t.device.type
        t_device = t.device.type
        assert (xt_device == self.device)
        assert (t_device == self.device)

        # Estimate the noise from (x_t,t)
        eps_theta = self.model(x_t, t)

        return eps_theta
    
    def train(self, n_epoch: int, batch_size: int, lr: float, dataset: Dataset, num_workers: int):
        '''
        Train the model which estimates the noise knowing x_t and t.

        Args:
            n_epoch (int): number of epoch(s) for the training.
            batch_size (int): dimension of a batch.
            lr (float): learning rate for the optimizer.
            dataset (torch.utils.data.Dataset): data on which we want to train the diffusion model.
            num_workers (int): number of workers for the dataloader
        '''
        # Instanciate a dataloader
        use_pin_memory = (self.device == "cuda")
        dataloader = DataLoader(dataset, batch_size, shuffle = True, num_workers = num_workers, pin_memory = use_pin_memory, drop_last = True)

        # Instanciate an optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = lr)

        # Loop on batches to optimize network parameters
        best = float('inf')
        loss_list = []

        for n in range(1, n_epoch + 1):

            tqdm_train = tqdm(dataloader, total = int(len(dataloader)))
            for _, x_0 in enumerate(tqdm_train):

                # Training mode
                x_0 = x_0.to(self.device)
                self.model.train()

                # Clean gradients
                optimizer.zero_grad()

                # Draw times
                times = self.draw_time(batch_size)

                # Draw noise
                noise = self.draw_noise(x_0)

                # Get x_t
                x_t = self.draw_xt(x_0, noise, times)

                # Predict eps_theta
                eps_theta = self.model(x_t, times)

                # Compute the loss
                loss = torch.nn.functional.mse_loss(eps_theta['sample'], noise, reduction = 'mean')

                # Update parameters
                loss.backward()
                optimizer.step()

                # Update lists
                loss_list.append(loss.item())
                tqdm_train.set_postfix(loss = loss.item())
