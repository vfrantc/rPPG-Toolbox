"""Quaternion Temporal Shift Convolutional Attention Network (Q-TS-CAN).
Quaternion version of Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class QuaternionTSCAN(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20, img_size=36):
        """Definition of Quaternion TS_CAN.
        Args:
          in_channels: the number of input channel. Default: 3
          frame_depth: the number of frame (window size) used in temport shift. Default: 20
          img_size: height/width of each frame. Default: 36.
        Returns:
          Quaternion TS_CAN model.
        """
        super(QuaternionTSCAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        # Quaternion layers need multiples of 4 channels
        self.nb_filters1 = nb_filters1 if nb_filters1 % 4 == 0 else ((nb_filters1 // 4) + 1) * 4
        self.nb_filters2 = nb_filters2 if nb_filters2 % 4 == 0 else ((nb_filters2 // 4) + 1) * 4
        self.nb_dense = nb_dense if nb_dense % 4 == 0 else ((nb_dense // 4) + 1) * 4
        
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        
        # Motion branch quaternion convs
        self.motion_conv1 = QuaternionConv(4, self.nb_filters1, kernel_size=self.kernel_size, 
                                          stride=1, padding=1, bias=True, operation='convolution2d')
        self.motion_conv2 = QuaternionConv(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, 
                                          stride=1, padding=0, bias=True, operation='convolution2d')
        self.motion_conv3 = QuaternionConv(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, 
                                          stride=1, padding=1, bias=True, operation='convolution2d')
        self.motion_conv4 = QuaternionConv(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, 
                                          stride=1, padding=0, bias=True, operation='convolution2d')
        
        # Appearance branch quaternion convs
        self.apperance_conv1 = QuaternionConv(4, self.nb_filters1, kernel_size=self.kernel_size,
                                             stride=1, padding=1, bias=True, operation='convolution2d')
        self.apperance_conv2 = QuaternionConv(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, 
                                             stride=1, padding=0, bias=True, operation='convolution2d')
        self.apperance_conv3 = QuaternionConv(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                             stride=1, padding=1, bias=True, operation='convolution2d')
        self.apperance_conv4 = QuaternionConv(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, 
                                             stride=1, padding=0, bias=True, operation='convolution2d')
        
        # Attention layers - these remain real-valued as they produce masks
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=0, bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=0, bias=True)
        self.attn_mask_2 = Attention_mask()
        
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        
        # Calculate the flattened size based on image size
        if img_size == 36:
            flattened_size = 3136
        elif img_size == 72:
            flattened_size = 16384
        elif img_size == 96:
            flattened_size = 30976
        elif img_size == 128:
            flattened_size = 57600
        else:
            raise Exception('Unsupported image size')
            
        # Adjust flattened size for quaternion channels
        flattened_size = (flattened_size // 64) * self.nb_filters2
        
        # Quaternion dense layers
        self.final_dense_1 = QuaternionLinearAutograd(flattened_size, self.nb_dense, bias=True)
        # Final layer outputs real value, so we use standard linear
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def _pad_to_quaternion(self, x):
        """Pad input to have 4 channels for quaternion processing"""
        if x.size(1) == 3:
            # Pad with zeros to make it 4 channels
            padding = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
            x = torch.cat([x, padding], dim=1)
        return x

    def _quaternion_activation(self, x):
        """Apply activation function suitable for quaternion networks"""
        # Split into quaternion components
        nb_hidden = x.size(1)
        r = x[:, :nb_hidden//4, :, :]
        i = x[:, nb_hidden//4:nb_hidden//2, :, :]
        j = x[:, nb_hidden//2:3*nb_hidden//4, :, :]
        k = x[:, 3*nb_hidden//4:, :, :]
        
        # Apply tanh to each component
        r = torch.tanh(r)
        i = torch.tanh(i)
        j = torch.tanh(j)
        k = torch.tanh(k)
        
        # Concatenate back
        return torch.cat([r, i, j, k], dim=1)

    def forward(self, inputs, params=None):
        # Split inputs
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]
        
        # Pad inputs to 4 channels for quaternion processing
        diff_input = self._pad_to_quaternion(diff_input)
        raw_input = self._pad_to_quaternion(raw_input)

        # Motion branch
        diff_input = self.TSM_1(diff_input)
        d1 = self._quaternion_activation(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = self._quaternion_activation(self.motion_conv2(d1))

        # Appearance branch
        r1 = self._quaternion_activation(self.apperance_conv1(raw_input))
        r2 = self._quaternion_activation(self.apperance_conv2(r1))

        # Attention mechanism (remains real-valued)
        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        # First pooling and dropout
        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        # Second motion branch
        d4 = self.TSM_3(d4)
        d5 = self._quaternion_activation(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = self._quaternion_activation(self.motion_conv4(d5))

        # Second appearance branch
        r5 = self._quaternion_activation(self.apperance_conv3(r4))
        r6 = self._quaternion_activation(self.apperance_conv4(r5))

        # Second attention
        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        # Final pooling and dense layers
        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        
        # Quaternion dense layer
        d10 = self._quaternion_activation_1d(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        
        # Final output (real-valued)
        out = self.final_dense_2(d11)

        return out
    
    def _quaternion_activation_1d(self, x):
        """Apply activation function for 1D quaternion tensors"""
        nb_hidden = x.size(1)
        r = x[:, :nb_hidden//4]
        i = x[:, nb_hidden//4:nb_hidden//2]
        j = x[:, nb_hidden//2:3*nb_hidden//4]
        k = x[:, 3*nb_hidden//4:]
        
        r = torch.tanh(r)
        i = torch.tanh(i)
        j = torch.tanh(j)
        k = torch.tanh(k)
        
        return torch.cat([r, i, j, k], dim=1)


class QuaternionMTTS_CAN(nn.Module):
    """Quaternion MTTS_CAN is the multi-task (respiration) version of Quaternion TS-CAN"""

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20):
        super(QuaternionMTTS_CAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        # Ensure filter counts are multiples of 4 for quaternion
        self.nb_filters1 = nb_filters1 if nb_filters1 % 4 == 0 else ((nb_filters1 // 4) + 1) * 4
        self.nb_filters2 = nb_filters2 if nb_filters2 % 4 == 0 else ((nb_filters2 // 4) + 1) * 4
        self.nb_dense = nb_dense if nb_dense % 4 == 0 else ((nb_dense // 4) + 1) * 4
        
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        
        # Motion branch quaternion convs
        self.motion_conv1 = QuaternionConv(4, self.nb_filters1, kernel_size=self.kernel_size,
                                          stride=1, padding=1, bias=True, operation='convolution2d')
        self.motion_conv2 = QuaternionConv(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size,
                                          stride=1, padding=0, bias=True, operation='convolution2d')
        self.motion_conv3 = QuaternionConv(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                          stride=1, padding=1, bias=True, operation='convolution2d')
        self.motion_conv4 = QuaternionConv(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size,
                                          stride=1, padding=0, bias=True, operation='convolution2d')
        
        # Appearance branch quaternion convs
        self.apperance_conv1 = QuaternionConv(4, self.nb_filters1, kernel_size=self.kernel_size,
                                             stride=1, padding=1, bias=True, operation='convolution2d')
        self.apperance_conv2 = QuaternionConv(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size,
                                             stride=1, padding=0, bias=True, operation='convolution2d')
        self.apperance_conv3 = QuaternionConv(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                             stride=1, padding=1, bias=True, operation='convolution2d')
        self.apperance_conv4 = QuaternionConv(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size,
                                             stride=1, padding=0, bias=True, operation='convolution2d')
        
        # Attention layers (remain real-valued)
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=0, bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=0, bias=True)
        self.attn_mask_2 = Attention_mask()
        
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4_y = nn.Dropout(self.dropout_rate2)
        self.dropout_4_r = nn.Dropout(self.dropout_rate2)

        # Adjusted flattened size for quaternion
        flattened_size = (16384 // 64) * self.nb_filters2
        
        # Quaternion dense layers
        self.final_dense_1_y = QuaternionLinearAutograd(flattened_size, self.nb_dense, bias=True)
        self.final_dense_2_y = nn.Linear(self.nb_dense, 1, bias=True)
        self.final_dense_1_r = QuaternionLinearAutograd(flattened_size, self.nb_dense, bias=True)
        self.final_dense_2_r = nn.Linear(self.nb_dense, 1, bias=True)

    def _pad_to_quaternion(self, x):
        """Pad input to have 4 channels for quaternion processing"""
        if x.size(1) == 3:
            padding = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
            x = torch.cat([x, padding], dim=1)
        return x

    def _quaternion_activation(self, x):
        """Apply activation function suitable for quaternion networks"""
        nb_hidden = x.size(1)
        r = x[:, :nb_hidden//4, :, :]
        i = x[:, nb_hidden//4:nb_hidden//2, :, :]
        j = x[:, nb_hidden//2:3*nb_hidden//4, :, :]
        k = x[:, 3*nb_hidden//4:, :, :]
        
        r = torch.tanh(r)
        i = torch.tanh(i)
        j = torch.tanh(j)
        k = torch.tanh(k)
        
        return torch.cat([r, i, j, k], dim=1)
    
    def _quaternion_activation_1d(self, x):
        """Apply activation function for 1D quaternion tensors"""
        nb_hidden = x.size(1)
        r = x[:, :nb_hidden//4]
        i = x[:, nb_hidden//4:nb_hidden//2]
        j = x[:, nb_hidden//2:3*nb_hidden//4]
        k = x[:, 3*nb_hidden//4:]
        
        r = torch.tanh(r)
        i = torch.tanh(i)
        j = torch.tanh(j)
        k = torch.tanh(k)
        
        return torch.cat([r, i, j, k], dim=1)

    def forward(self, inputs, params=None):
        # Split inputs
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]
        
        # Pad inputs to 4 channels for quaternion processing
        diff_input = self._pad_to_quaternion(diff_input)
        raw_input = self._pad_to_quaternion(raw_input)

        # Motion branch
        diff_input = self.TSM_1(diff_input)
        d1 = self._quaternion_activation(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = self._quaternion_activation(self.motion_conv2(d1))

        # Appearance branch
        r1 = self._quaternion_activation(self.apperance_conv1(raw_input))
        r2 = self._quaternion_activation(self.apperance_conv2(r1))

        # Attention mechanism
        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        # First pooling and dropout
        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        # Second motion branch
        d4 = self.TSM_3(d4)
        d5 = self._quaternion_activation(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = self._quaternion_activation(self.motion_conv4(d5))

        # Second appearance branch
        r5 = self._quaternion_activation(self.apperance_conv3(r4))
        r6 = self._quaternion_activation(self.apperance_conv4(r5))

        # Second attention
        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        # Final pooling
        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)

        # Branch for primary output
        d10 = self._quaternion_activation_1d(self.final_dense_1_y(d9))
        d11 = self.dropout_4_y(d10)
        out_y = self.final_dense_2_y(d11)

        # Branch for secondary output
        d10 = self._quaternion_activation_1d(self.final_dense_1_r(d9))
        d11 = self.dropout_4_r(d10)
        out_r = self.final_dense_2_r(d11)

        return out_y, out_r