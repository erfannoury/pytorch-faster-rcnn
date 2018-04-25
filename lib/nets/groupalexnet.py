# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from collections import OrderedDict


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from nets.network import Network
from model.config import cfg
from nets.utils import Lambda, CustomSpatialCrossMapLRN


class TorchAlexNetWithGrouping(nn.Module):
    """
    This is the BVLC CaffeNet model with optional batch normalization
    """

    def __init__(self, num_classes=21, add_bn=False):
        super(TorchAlexNetWithGrouping, self).__init__()

        if add_bn:
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 96, (11, 11), (4, 4))),
                ('batchnorm1', nn.BatchNorm2d(96)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d((3, 3), (2, 2), (0, 0),
                                       ceil_mode=True)),
                ('conv2', nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2), 1, 2)),
                ('batchnorm2', nn.BatchNorm2d(256)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d((3, 3), (2, 2), (0, 0),
                                       ceil_mode=True)),
                ('conv3', nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1))),
                ('batchnorm3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),
                ('conv4', nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1), 1, 2)),
                ('batchnorm4', nn.BatchNorm2d(384)),
                ('relu4', nn.ReLU(inplace=True)),
                ('conv5', nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1), 1, 2)),
                ('batchnorm5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('pool5', nn.MaxPool2d((3, 3), (2, 2), (0, 0),
                                       ceil_mode=True)),
            ]))

            self.classifier = nn.Sequential(OrderedDict([
                ('drop5', nn.Dropout()),
                ('fc6', nn.Linear(256 * 6 * 6, 4096)),
                ('batchnorm6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
                ('drop6', nn.Dropout()),
                ('fc7', nn.Linear(4096, 4096)),
                ('batchnorm7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc8', nn.Linear(4096, num_classes)),
            ]))

        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 96, (11, 11), (4, 4))),
                ('relu1', nn.ReLU(inplace=True)),
                ('lrn1', Lambda(lambda x, lrn=CustomSpatialCrossMapLRN(
                    *(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data)))),
                ('pool1', nn.MaxPool2d((3, 3), (2, 2), (0, 0),
                                       ceil_mode=True)),
                ('conv2', nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2), 1, 2)),
                ('relu2', nn.ReLU(inplace=True)),
                ('lrn2', Lambda(lambda x, lrn=CustomSpatialCrossMapLRN(
                    *(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data)))),
                ('pool2', nn.MaxPool2d((3, 3), (2, 2), (0, 0),
                                       ceil_mode=True)),
                ('conv3', nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1))),
                ('relu3', nn.ReLU(inplace=True)),
                ('conv4', nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1), 1, 2)),
                ('relu4', nn.ReLU(inplace=True)),
                ('conv5', nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1), 1, 2)),
                ('relu5', nn.ReLU(inplace=True)),
                ('pool5', nn.MaxPool2d((3, 3), (2, 2), (0, 0),
                                       ceil_mode=True)),
            ]))

            self.classifier = nn.Sequential(OrderedDict([
                ('drop5', nn.Dropout()),
                ('fc6', nn.Linear(256 * 6 * 6, 4096)),
                ('relu6', nn.ReLU(inplace=True)),
                ('drop6', nn.Dropout()),
                ('fc7', nn.Linear(4096, 4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc8', nn.Linear(4096, num_classes)),
            ]))

    def load(self, checkpoint_file):
        current_state_dict = self.state_dict()
        old_state = torch.load(checkpoint_file)
        new_state = {}
        for idx, i in enumerate([0, 4, 8, 10, 12]):
            for name in ['weight', 'bias']:
                w = np.array(old_state[f'{i}.{name}'])
                new_state[f'features.conv{idx+1}.{name}'] = \
                    torch.from_numpy(w).cuda()

        current_state_dict.update(new_state)
        self.load_state_dict(current_state_dict)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class groupalexnet(Network):
    def __init__(self):
        Network.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._net_conv_channels = 256
        self._fc7_channels = 4096

    def _init_head_tail(self):
        self.alexnet = TorchAlexNetWithGrouping(
            add_bn=cfg.GROUPALEXNET.ADD_BN)
        # Remove fc8
        self.alexnet.classifier = nn.Sequential(
            *list(self.alexnet.classifier._modules.values())[:-1])

        # # Fix the layers before conv3:
        # for layer in range(6):
        #     for p in self.alexnet.features[layer].parameters():
        #         p.requires_grad = False

        # not using the last maxpool layer
        self._layers['head'] = nn.Sequential(
            *list(self.alexnet.features._modules.values())[:-1])

    def _image_to_head(self):
        net_conv = self._layers['head'](self._image)
        self._act_summaries['conv'] = net_conv

        return net_conv

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.alexnet.classifier(pool5_flat)

        return fc7

    def load_pretrained_cnn(self, state_dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        new_state = {}
        for idx, i in enumerate([0, 4, 8, 10, 12]):
            for name in ['weight', 'bias']:
                w = np.array(state_dict[f'{i}.{name}'])
                new_state[f'features.conv{idx+1}.{name}'] = \
                    torch.from_numpy(w).cuda()
        if cfg.GROUPALEXNET.LOAD_FC:
            for idx, i in enumerate([16, 19]):
                for name in ['weight', 'bias']:
                    w = np.array(state_dict[f'{i}.1.{name}'])
                    new_state[f'classifier.fc{idx+6}.{name}'] = \
                        torch.from_numpy(w).cuda()

        current_state_dict = self.alexnet.state_dict()
        current_state_dict.update(new_state)
        self.alexnet.load_state_dict(current_state_dict)
