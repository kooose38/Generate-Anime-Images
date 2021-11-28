from torchvision import transforms 
from PIL import Image 

from backend.gan.constant import CFG

def get_detect_dataset(img: Image.Image, config: CFG):
   transform = transforms.Compose([
      transforms.Resize((config.img_size, config.img_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=config.mean, std=config.std)
   ])
   return transform(img)