from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn
from config import config
import torch
from collections import OrderedDict
import torch.nn.functional as F

def get_net():
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    if config.channels == 4:
        w = model.conv1_7x7_s2.weight
        model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        model.conv1_7x7_s2.weight = nn.Parameter(torch.cat((w, (w[:,:1,:,:] + w[:,1:2,:,:])/2), dim=1))
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, config.num_classes, bias=True),
            )
    return model
