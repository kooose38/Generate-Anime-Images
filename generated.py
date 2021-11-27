import torch 
import json
import os 
import matplotlib.pyplot as plt 

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
         fake_img = fake_img.permute(1, 2, 0).detach().numpy() # [W, H, C]

      saved_img(fake_img, uid)
      return json.dumps({
         "status_code": 200, 
         "saved_img_path": f"templates/images/{uid}.png"
      })
   
   except Exception as e:
      return json.dumps({
         "status_code": 503, 
         "message": e
      })


   




