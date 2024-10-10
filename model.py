import torch
import torch.nn as nn
import torchvision
from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class L2CS(models.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(L2CS, self).__init__(block, layers, num_classes)
        self.fc = Identity()
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_classes)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        # gaze
        pre_yaw_gaze = self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze