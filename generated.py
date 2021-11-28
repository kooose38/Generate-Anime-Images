import torch 
import json
import os 
import matplotlib.pyplot as plt 
from PIL import Image 

from backend.gan.constant import CFG
from backend.gan.dataset import create_dataset
from backend.gan.net import Generator

def deleted_img():
   for path in os.listdir("templates/images"):
      os.remove(os.path.join("templates/images", path))

def load_net(config):
   net = Generator(config)
   if config.debug is not True:
      net.load_state_dict(
         torch.load(config.model_path, map_location={"cuda:0": "cpu"})
      )
   net.eval()
   return net 

def saved_img(img, uid):
   fig = plt.figure()
   plt.imshow(img)
   plt.axis("off")
   fig.savefig(f"templates/images/{uid}.png", dpi=150)


def generated_img(uid):
   try: 
      deleted_img()
      config = CFG()
      net = load_net(config)
      noise = create_dataset(config)

      with torch.no_grad():
         fake_img = net(noise).squeeze(0)
         fake_img_ = fake_img.permute(1, 2, 0).detach().numpy() # [W, H, C]

      saved_img(fake_img_, uid)
      print(fake_img.size())
      return Image.open(f"templates/images/{uid}.png").convert("RGB")
   
   except Exception as e:
      print(f"[ERROR]{e}")
      return False


   




