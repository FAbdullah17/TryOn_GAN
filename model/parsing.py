import torch
import torch.nn as nn
import torchvision.models as models

class ParsingModule(nn.Module):
    """ Segment human body into different regions (skin, clothes, etc.). """
    def __init__(self):
        super(ParsingModule, self).__init__()
        self.segmentation_net = models.resnet18(pretrained=True)
        self.segmentation_net.fc = nn.Linear(512, 7)  # Predicts 7 classes

    def forward(self, person_img):
        """ Predicts segmented regions of the human body """
        return self.segmentation_net(person_img)
