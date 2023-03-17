import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model = models.resnet18()
        # self.model.fc = nn.Linear(512 * 4, output_dim)
        self.model.conv1 = nn.Conv2d(3, 64, (7,7), (2,2), (3,3))
        self.model.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        # if x.size(1) == 1:
        #     x = torch.cat([x, x, x], dim=1)
        out = self.model(x)

        return out


class CrossEntropy(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.model = models.resnet18()
        # self.model.fc = nn.Linear(512 * 4, output_dim)
        self.model.conv1 = nn.Conv2d(3, 64, (7,7), (2,2), (3,3))
        self.model.fc = nn.Sequential(nn.ReLU(),(nn.Linear(512, classes))) # Classify

    def forward(self, x):
        # if x.size(1) == 1:
        #     x = torch.cat([x, x, x], dim=1)
        out = self.model(x)

        return out