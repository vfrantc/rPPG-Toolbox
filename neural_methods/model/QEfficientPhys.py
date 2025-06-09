"""Quaternion EfficientPhys: Quaternion-based version of EfficientPhys
Based on the original EfficientPhys paper but using quaternion neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from neural_methods.model.quaternion_layers import QuaternionConv, QuaternionLinearAutograd


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


class QuaternionEfficientPhys(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20, img_size=36, channel='raw'):
        super(QuaternionEfficientPhys, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        # Quaternion layers expect channels divisible by 4
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        
        # TSM layers - these work on quaternion feature maps as well
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        
        # Motion branch quaternion convs
        # Input will be padded from 3 to 4 channels
        self.motion_conv1 = QuaternionConv(4, self.nb_filters1, kernel_size=self.kernel_size, 
                                          stride=1, padding=1, bias=True, 
                                          init_criterion='glorot', weight_init='quaternion',
                                          operation='convolution2d', rotation=False, 
                                          quaternion_format=True)
        
        self.motion_conv2 = QuaternionConv(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, 
                                          stride=1, padding=0, bias=True,
                                          init_criterion='glorot', weight_init='quaternion',
                                          operation='convolution2d', rotation=False, 
                                          quaternion_format=True)
        
        self.motion_conv3 = QuaternionConv(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, 
                                          stride=1, padding=1, bias=True,
                                          init_criterion='glorot', weight_init='quaternion',
                                          operation='convolution2d', rotation=False, 
                                          quaternion_format=True)
        
        self.motion_conv4 = QuaternionConv(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, 
                                          stride=1, padding=0, bias=True,
                                          init_criterion='glorot', weight_init='quaternion',
                                          operation='convolution2d', rotation=False, 
                                          quaternion_format=True)
        
        # Attention layers - these process quaternion features to produce real attention maps
        # We use regular Conv2d for attention since we want scalar attention weights
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=0, bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=0, bias=True)
        self.attn_mask_2 = Attention_mask()
        
        # Avg pooling - works the same for quaternion features
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        
        # Dense layers - using quaternion linear layers
        if img_size == 36:
            dense_input_size = 3136
        elif img_size == 72:
            dense_input_size = 16384
        elif img_size == 96:
            dense_input_size = 30976
        else:
            raise Exception('Unsupported image size')
            
        # Ensure dense sizes are divisible by 4 for quaternion layers
        self.nb_dense_quaternion = ((self.nb_dense + 3) // 4) * 4
        
        self.final_dense_1 = QuaternionLinearAutograd(dense_input_size, self.nb_dense_quaternion, 
                                                     bias=True, init_criterion='glorot',
                                                     weight_init='quaternion', rotation=False,
                                                     quaternion_format=True)
        
        # Output layer - regular linear since we want scalar output
        self.final_dense_2 = nn.Linear(self.nb_dense_quaternion, 1, bias=True)
        
        # Batch norm for input - we'll apply it before padding
        self.batch_norm = nn.BatchNorm2d(3)
        self.channel = channel

    def pad_to_quaternion(self, x):
        """Pad 3-channel input to 4 channels by adding zeros"""
        if x.size(1) == 3:
            # Add zero channel to make it 4 channels
            zero_channel = torch.zeros(x.size(0), 1, x.size(2), x.size(3), 
                                     device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_channel], dim=1)
        return x

    def forward(self, inputs, params=None):
        # Compute frame differences
        inputs = torch.diff(inputs, dim=0)
        
        # Apply batch norm to 3-channel input
        inputs = self.batch_norm(inputs)
        
        # Pad to 4 channels for quaternion processing
        inputs = self.pad_to_quaternion(inputs)

        # Motion branch with quaternion operations
        network_input = self.TSM_1(inputs)
        d1 = torch.tanh(self.motion_conv1(network_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        # Attention mechanism - produces scalar attention maps
        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        # Apply attention to quaternion features
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        # Second attention mechanism
        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        
        # Flatten for dense layers
        d9 = d8.view(d8.size(0), -1)
        
        # Ensure input size is divisible by 4 for quaternion linear layer
        if d9.size(1) % 4 != 0:
            padding_size = 4 - (d9.size(1) % 4)
            padding = torch.zeros(d9.size(0), padding_size, device=d9.device, dtype=d9.dtype)
            d9 = torch.cat([d9, padding], dim=1)
        
        # Quaternion dense layer
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        
        # Final output layer (regular linear for scalar output)
        out = self.final_dense_2(d11)

        return out


# Helper function to create the model
def create_quaternion_efficientphys(in_channels=3, nb_filters1=32, nb_filters2=64, 
                                   kernel_size=3, dropout_rate1=0.25, dropout_rate2=0.5, 
                                   pool_size=(2, 2), nb_dense=128, frame_depth=20, 
                                   img_size=36, channel='raw'):
    """
    Create a Quaternion EfficientPhys model
    
    Args:
        in_channels: Number of input channels (will be padded to 4 if needed)
        nb_filters1: Number of filters in first conv layers (must be divisible by 4)
        nb_filters2: Number of filters in second conv layers (must be divisible by 4)
        kernel_size: Size of convolutional kernels
        dropout_rate1: Dropout rate for first set of dropout layers
        dropout_rate2: Dropout rate for final dropout layer
        pool_size: Size of pooling windows
        nb_dense: Number of units in dense layer (will be adjusted to be divisible by 4)
        frame_depth: Number of frames for TSM
        img_size: Input image size (36, 72, or 96)
        channel: Channel type ('raw' or other)
    
    Returns:
        QuaternionEfficientPhys model
    """
    # Ensure filter sizes are divisible by 4 for quaternion layers
    nb_filters1 = ((nb_filters1 + 3) // 4) * 4
    nb_filters2 = ((nb_filters2 + 3) // 4) * 4
    
    return QuaternionEfficientPhys(
        in_channels=in_channels,
        nb_filters1=nb_filters1,
        nb_filters2=nb_filters2,
        kernel_size=kernel_size,
        dropout_rate1=dropout_rate1,
        dropout_rate2=dropout_rate2,
        pool_size=pool_size,
        nb_dense=nb_dense,
        frame_depth=frame_depth,
        img_size=img_size,
        channel=channel
    )