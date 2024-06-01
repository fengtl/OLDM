# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class FCDiscriminator(nn.Module):
    """
    inplanes, planes. Patch-gan
    """

    def __init__(self, inplanes, planes = 64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(planes*8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class FCDiscriminator_low(nn.Module):
    """
    inplanes, planes. Patch-gan
    """

    def __init__(self, inplanes, planes = 64):
        super(FCDiscriminator_low, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(planes*4, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class FCDiscriminator_out(nn.Module):
    """
    inplanes, planes. Patch-gan
    """

    def __init__(self, inplanes, planes = 64):
        super(FCDiscriminator_out, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(planes*8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class FCDiscriminator_class(nn.Module): #TODO: whether reduce channels before pooling, whether update pred, more complex discriminator
                                        #TODO: 19 different discriminators or 1 discriminator after projection
    """
    inplanes, planes. gan
    """
    class DISCRIMINATOR(nn.Module):
        def __init__(self,inplanes, planes=64):
            super(FCDiscriminator_class.DISCRIMINATOR, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.classifier = nn.Conv2d(planes*8, 1, kernel_size=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.leaky_relu(x)
            x = self.classifier(x)
            return x
            
    def __init__(self, inplanes, midplanes, planes = 32):
        '''
        midplanes: channel size after reduction
        '''
        super(FCDiscriminator_class, self).__init__()
        self.inplace = inplanes
        self.midplanes = midplanes
        self.planes = planes
        self.source_unique = []
        self.target_unique = []
        self.common_unique = []
        self.discriminator = self.DISCRIMINATOR(inplanes)

    def forward(self, x):
        x = self.discriminator(x)
        pass
        return x

    def calc_common_unique(self, source_unique, target_unique):
        self.source_unique = source_unique
        self.target_unique = target_unique
        self.common_unique = []
        for i in range(19):
            if (i in self.source_unique) and (i in self.target_unique):
                self.common_unique.append(i)
        pass

    def calc_valid_unique(self, classes_list):
        self.valid_unique = []
        for i in range(19):
            if (i in classes_list):
                self.valid_unique.append(i)