from collections import namedtuple

import torch
from torch import nn
from torchvision import models


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.weights = models.VGG19_Weights.DEFAULT
        self.vgg_19_features = models.vgg19(weights=self.weights).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), self.vgg_19_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), self.vgg_19_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), self.vgg_19_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), self.vgg_19_features[x])
        for x in range(21, 22):
            self.slice5.add_module(str(x), self.vgg_19_features[x])
        for x in range(22, 30):
            self.slice6.add_module(str(x), self.vgg_19_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        layer1_1 = self.slice1(x)
        layer2_1 = self.slice2(layer1_1)
        layer3_1 = self.slice3(layer2_1)
        layer4_1 = self.slice4(layer3_1)
        conv4_2 = self.slice5(layer4_1)
        layer5_1 = self.slice6(conv4_2)
        return layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1

    def get_transform(self):
        return self.weights.transforms()
