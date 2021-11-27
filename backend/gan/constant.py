import os 

class CFG:
   debug = False 
   batch_size = 1 
   img_size = 64 
   z_fill = 100
   mid_size = 64
   n_channel = 3 
   model_path = os.path.join("models/", "anime_G_50.pth")
