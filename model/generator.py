import torch
import torch.nn as nn
import torchvision.models as models

class Generator(nn.Module):
    """ Generates a new image of the person wearing the warped clothing. """
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, parsed_human, warped_clothing):
        """ Combines parsed human and warped clothing for final output """
        combined = torch.cat([parsed_human, warped_clothing], dim=1)
        encoded = self.encoder(combined)
        output = self.decoder(encoded.view(-1, 512, 1, 1))
        return output
