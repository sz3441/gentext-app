import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import make_grid

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 4, stride = 2, padding = 1)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.act2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x



class Generator(nn.Module):
    def __init__(self, z_dim = 100):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.reshape = lambda x: x.view(x.size(0), 128, 7, 7)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding = 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.tanh(x)

        return x