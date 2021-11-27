import torch 
from torchvision import transforms 
from torch.utils.data import Dataset 
from PIL import Image 

from backend.gan.constant import CFG

class AnimeDetectDataset(Dataset):
   def __init__(self, f: str, config: CFG):
      self.file = list(f) 
      self.transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)       
        ])

   def __getitem__(self, idx):
      img = Image.open(self.file[idx]).convert("RGB")
      img = self.transform(img)
      return img 

   def __len__(self):
      return len(self.file)