import json 
import torch 

from backend.vgg.constant import CFG
from backend.vgg.dataset import get_detect_dataset
from backend.vgg.net import AnimeDetectModel

def load_model(n_classes, config):
   net = AnimeDetectModel(n_classes)
   if config.debug is not True:
      net.load_state_dict(
         torch.load(
            config.model_path, map_location={"cuda:0": "cpu"}
         )
      )
   net.eval()
   print("[INFO] success loading for detect model!")
   return net 


def load_label(file):
   with open(file) as f:
      label = json.load(f)
   return {int(k): v for k, v in label.items()}


def detect(img):
   config = CFG()
   index2label = load_label(config.label_path)
   net = load_model(len(index2label), config)

   ds = get_detect_dataset(img, config)
   dl = ds.unsqueeze(0) # [batch, C, W, H]

   with torch.no_grad():
      output = net(dl)
      output = torch.nn.Softmax(dim=1)(output)
      pred = output.topk(5)[0][0].detach().cpu().numpy().tolist() # confidence 
      pred_id = output.topk(5)[1][0].detach().cpu().numpy().tolist() # predict index 

      results = []
      for p, idx in zip(pred, pred_id):
         result = {
            "score": round(p, 2) * 100, 
            "predict": index2label[int(idx)]
         }
         results.append(result)

   return results 
