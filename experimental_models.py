import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.nn import init
import torchvision.models as models
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class ResNetModified(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ResNetModified, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in self.resnet.parameters():
            param.requires_grad = False

        for params in self.resnet.fc.parameters():
            params.requires_grad = True

        for params in self.resnet.conv1.parameters():
            params.requires_grad = True

        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(
            5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNetExtended(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ResNetExtended, self).__init__()
        self.preprocessing_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='nearest')
        )

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in self.resnet.parameters():
            param.requires_grad = False

        for params in self.resnet.fc.parameters():
            params.requires_grad = True

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

    def forward(self, x):
        x = self.preprocessing_layer(x)
        x = self.resnet(x)
        return x


class ResNetMLP(nn.Module):
    def __init__(self, base_model_resnet, n_bands_res, input_mlp, n_bands_mlp, n_class, p_drop=0.3):
        super().__init__()
        self.resnet = HeadlessResnet(base_model_resnet,
                                     n_class,
                                     n_bands_res,
                                     p_drop)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.mlp = FullyConnectedNetwork(input_mlp,
                                         n_bands_mlp,
                                         p_drop)

        self.head = nn.Sequential(
            nn.Linear(512+512, 2048),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.Linear(2048, n_class)

        )


class HeadlessResnet(nn.Module):
    def __init__(self, pretrained_resnet, n_classes, n_bands=30, p_dropout=0.3):
        super().__init__(base_model=pretrained_resnet,
                         n_classes=n_classes,
                         n_bands=n_bands,
                         p_dropout=p_dropout,
                         headless=True)


class FullyConnectedNetwork(nn.Module):
    """Multi-layer perceptron model. Bascially just uses fully connected
    layers rather than 3x3 convs. Only can work with very small images,
    otherwise the number of weights in the model will be way too high given
    that each layer is fully connected.
    """

    def __init__(self, input_size, n_bands, p_drop=0.3, n_class=0):
        super().__init__()

        self.fc1 = nn.Linear(input_size*input_size*n_bands, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)

        self.dropout = nn.Dropout(p_drop)

        # set n_class to 0 if we want headless model
        self.n_class = n_class
        if n_class:
            self.head = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(p=p_drop),
                nn.Linear(1024, n_class)
            )

    def forward(self, x):
        # flatten image input
        _, c, h, w = x.shape
        x = x.view(-1, c*h*w)  # [batch, c*h*w]

        x = F.relu(self.fc1(x))  # [batch, 512]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # [batch, 512]
        x = self.dropout(x)
        x = F.relu(self.fc3(x))  # [batch, 512]
        x = self.dropout(x)
        x = F.relu(self.fc4(x))  # [batch, 512]

        if self.n_class:
            x = self.head(x)

        return x


class Expand(torch.nn.Module):
    def __init__(self, in_channels, e1_out_channels, e3_out_channels):
        super(Expand, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, e1_out_channels, (1, 1))
        self.conv_3x3 = nn.Conv2d(
            in_channels, e3_out_channels, (3, 3), padding=1)

    def forward(self, x):
        o1 = self.conv_1x1(x)
        o3 = self.conv_3x3(x)
        return torch.cat((o1, o3), dim=1)


class DepthSepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthSepConv, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, groups=in_channels, padding=1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class Fire(nn.Module):
    """
      Fire module in SqueezeNet
      out_channles = e1x1 + e3x3
      Eg.: input: ?xin_channelsx?x?
           output: ?x(e1x1+e3x3)x?x?
    """

    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(Fire, self).__init__()

        # squeeze
        self.squeeze = nn.Conv2d(in_channels, s1x1, (1, 1))
        self.sq_act = nn.LeakyReLU(0.1)

        # expand
        self.expand = Expand(s1x1, e1x1, e3x3)
        self.ex_act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.sq_act(self.squeeze(x))
        x = self.ex_act(self.expand(x))
        return x


class SpectrumNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SpectrumNet, self).__init__()
        self.conv_1 = DepthSepConv(in_channels, 32)
        self.spectral_2 = Fire(32, 16, 96, 32)
        self.spectral_3 = Fire(128, 16, 96, 32)
        self.spectral_4 = Fire(128, 32, 192, 64)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spectral_5 = Fire(256, 48, 192, 64)
        self.spectral_6 = Fire(256, 48, 288, 96)
        self.spectral_7 = Fire(384, 48, 288, 96)
        self.spectral_8 = Fire(384, 64, 384, 128)
        self.maxpool_8 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spectral_9 = Fire(512, 64, 384, 128)
        self.conv_10 = DepthSepConv(512, 10)
        self.avgpool_10 = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.spectral_2(x)
        x = self.spectral_3(x)
        x = self.spectral_4(x)
        x = self.maxpool_4(x)
        x = self.spectral_5(x)
        x = self.spectral_6(x)
        x = self.spectral_7(x)
        x = self.spectral_8(x)
        x = self.maxpool_8(x)
        x = self.spectral_9(x)
        x = self.conv_10(x)
        x = self.avgpool_10(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class MyNet(nn.Module):
    def __init__(self, num_classes):
        super(MyNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.dense_layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.bottleneck(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dense_layers(x)

        return x

# best performance: 24.5                                                                                                                                                                                                                      %
