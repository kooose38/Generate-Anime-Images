import os 

class CFG:
   debug = False 
   img_size = 224 
   mean = (.485, .456, .406)
   std = (.229, .224, .225)
   n_channel = 3 
   batch_size = 1 
   model_path = os.path.join("models", "detect_10.pth")
   label_path = "label.json"



