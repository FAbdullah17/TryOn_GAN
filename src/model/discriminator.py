import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_features = 64
        for _ in range(3):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_features = out_features

        model += [nn.Conv2d(in_features, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
