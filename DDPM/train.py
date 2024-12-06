# Libraries
import utils
import torch
import pickle
import diffusion
import diffusers

# Model parameters
img_channels = 1
base_channels = 128
time_emb_dim = 128
attention_resolution = (1,)

# Diffusion parameters
T = 4000
input_shape = (1,32,32)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
n_epochs = 30
batch_size = 128
lr = 5e-5
dataset = utils.MNIST()
num_workers = 4

# Instanciate a neural network architecture and a diffusion model
unet_architecture = diffusers.UNet2DModel(sample_size = (32,32), in_channels = 1, out_channels = 1, block_out_channels = (64, 128, 256, 512))
diffusion_model = diffusion.DiffusionModel(unet_architecture, input_shape, device, T = 4000, scheduler = 'cosine')

# Train the model
diffusion_model.train(n_epochs, batch_size, lr, dataset, num_workers)

# Save the model after training
with open('./trained_diffusion.pkl', 'wb') as file:
    pickle.dump(diffusion_model, file)