import torch 

def create_dataset(config):
   z = torch.randn(1, config.z_fill, 1, 1)
   return z 
