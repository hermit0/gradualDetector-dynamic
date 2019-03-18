# -*- coding: utf-8 -*-
#TALNN

import torch
import torch.nn as nn
import math

__all__ = ['TALNN']

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

def conv1x1x1(in_planes, out_planes, stride=1):
    # 1x1x1 convolution
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False)

class Head(nn.Module):
    def __init__(self,inplanes,avg_size,avg_length):
        super(Head, self).__init__()
        self.conv = conv1x1x1(inplanes,1,stride=1)
        self.avgpool = nn.AvgPool3d((avg_length,avg_size,avg_size),stride=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):    #不修改输入
        out = self.conv(x)
        out = self.avgpool(out)
        out = self.sigmoid(out)
        out = out
        return out

class TALNN(nn.Module):
    def __init__(self,sample_size,sample_duration):
        super(TALNN, self).__init__()
        self.layer1 = BasicBlock(3,32,stride=2)
        self.layer2 = BasicBlock(32,64,stride = 2)
        self.layer3 = BasicBlock(64,128,stride = 2)
        self.layer4 = BasicBlock(128,256,stride = 2)
        
        last_duration = int(math.ceil(sample_duration/16))
        last_size = int(math.ceil(sample_size / 16) )
        
        self.head1 = Head(256,last_size,last_duration)
        self.head2 = Head(256,last_size,last_duration)
        self.head3 = Head(256,last_size,last_duration)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        head1_output = self.head1(x)
        head1_output = head1_output.view(x.size(0),-1)
        head2_output = self.head2(x)
        head2_output = head2_output.view(x.size(0),-1)
        head3_output = self.head3(x)
        head3_output = head3_output.view(x.size(0),-1)
        out_list = (head1_output,head2_output,head3_output)
        
        return out_list

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('head1')
    ft_module_names.append('head2')
    ft_module_names.append('head3')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break

    return parameters