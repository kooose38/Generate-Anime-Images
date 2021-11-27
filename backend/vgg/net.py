import torch.nn as nn 
from torchvision.models import vgg19

class AnimeDetectModel(nn.Module):
    def __init__(self, n_classes):
        super(AnimeDetectModel, self).__init__()
        self.vgg = vgg19(pretrained=False)
        # for w in self.vgg.parameters():
        #     w.requires_grad = False 
        self.fc = nn.Linear(1000, n_classes)

    def forward(self, x):
        y = self.vgg(x)
        y = self.fc(y)
        return y 