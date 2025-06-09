"""
Quaternion PhysNet
Modified version of PhysNet using quaternion-valued layers
Maintains same input/output dimensionality as original
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
# Import the quaternion layers from the provided library
from neural_methods.model.quaternion_layers import QuaternionConv, QuaternionLinearAutograd
from neural_methods.model.quaternion_ops import get_r, get_i, get_j, get_k


class QuaternionBatchNorm3d(nn.Module):
    """Custom Quaternion BatchNorm3d that processes quaternion components separately"""
    def __init__(self, num_features):
        super(QuaternionBatchNorm3d, self).__init__()
        # Create separate batch norm for each quaternion component
        self.bn_r = nn.BatchNorm3d(num_features // 4)
        self.bn_i = nn.BatchNorm3d(num_features // 4)
        self.bn_j = nn.BatchNorm3d(num_features // 4)
        self.bn_k = nn.BatchNorm3d(num_features // 4)
    
    def forward(self, x):
        # Split into quaternion components
        r = get_r(x)
        i = get_i(x)
        j = get_j(x)
        k = get_k(x)
        
        # Apply batch norm to each component
        r = self.bn_r(r)
        i = self.bn_i(i)
        j = self.bn_j(j)
        k = self.bn_k(k)
        
        # Concatenate back
        return torch.cat([r, i, j, k], dim=1)


class QuaternionPhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(QuaternionPhysNet_padding_Encoder_Decoder_MAX, self).__init__()
        
        # Input alignment layer - converts 3 channels to 4 for quaternion processing
        self.input_align = nn.Conv3d(3, 4, [1, 1, 1], stride=1, padding=0, bias=False)
        
        # Quaternion Convolution Blocks
        self.ConvBlock1 = nn.Sequential(
            QuaternionConv(4, 16, [1, 5, 5], stride=1, padding=[0, 2, 2], 
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            QuaternionConv(16, 32, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock3 = nn.Sequential(
            QuaternionConv(32, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock5 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock6 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock7 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock8 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock9 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Upsample layers using regular ConvTranspose3d 
        # (quaternion transpose conv would need custom implementation)
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, 
                             kernel_size=[4, 1, 1], stride=[2, 1, 1], 
                             padding=[1, 0, 0]),
            QuaternionBatchNorm3d(64),
            nn.ELU(),
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, 
                             kernel_size=[4, 1, 1], stride=[2, 1, 1], 
                             padding=[1, 0, 0]),
            QuaternionBatchNorm3d(64),
            nn.ELU(),
        )

        # Final output layer - regular conv to get single channel output
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        # Pooling layers
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        
        # Initialize input alignment with small random values
        nn.init.xavier_uniform_(self.input_align.weight)

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape
        
        # Align input from 3 channels to 4 channels for quaternion processing
        x = self.input_align(x)  # [batch, 4, T, 128, 128]
        
        x = self.ConvBlock1(x)  # x [16, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [64, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)  # x [64, T/2, 32,32]

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        x = self.poolspa(x)  # x [64, T, 1,1]
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, length)

        return rPPG, x_visual, x_visual3232, x_visual1616


# Alternative implementation with quaternion transpose convolution
class QuaternionPhysNet_Full(nn.Module):
    """Full quaternion implementation including quaternion transpose convolution"""
    def __init__(self, frames=128):
        super(QuaternionPhysNet_Full, self).__init__()
        
        # Input alignment layer
        self.input_align = nn.Conv3d(3, 4, [1, 1, 1], stride=1, padding=0, bias=False)
        
        # Quaternion Convolution Blocks
        self.ConvBlock1 = nn.Sequential(
            QuaternionConv(4, 16, [1, 5, 5], stride=1, padding=[0, 2, 2], 
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            QuaternionConv(16, 32, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock3 = nn.Sequential(
            QuaternionConv(32, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Encoder blocks
        self.ConvBlock4 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock5 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock6 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock7 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock8 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock9 = nn.Sequential(
            QuaternionConv(64, 64, [3, 3, 3], stride=1, padding=1,
                          operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Quaternion transpose convolution for upsampling
        from quaternion_layers import QuaternionTransposeConv
        
        self.upsample = nn.Sequential(
            QuaternionTransposeConv(64, 64, kernel_size=[4, 1, 1], 
                                   stride=[2, 1, 1], padding=[1, 0, 0],
                                   operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ELU(),
        )
        
        self.upsample2 = nn.Sequential(
            QuaternionTransposeConv(64, 64, kernel_size=[4, 1, 1], 
                                   stride=[2, 1, 1], padding=[1, 0, 0],
                                   operation='convolution3d', quaternion_format=True),
            QuaternionBatchNorm3d(64),
            nn.ELU(),
        )

        # Output projection from quaternion space to single channel
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        # Pooling layers
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        
        # Initialize alignment layer
        nn.init.xavier_uniform_(self.input_align.weight)

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape
        
        # Align input from 3 channels to 4 channels
        x = self.input_align(x)  # [batch, 4, T, 128, 128]
        
        # Encoder path
        x = self.ConvBlock1(x)  # x [16, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [64, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)  # x [64, T/2, 32,32]

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        # Bottleneck
        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        
        # Decoder path
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        # Final processing
        x = self.poolspa(x)  # x [64, T, 1,1]
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, length)

        return rPPG, x_visual, x_visual3232, x_visual1616


# Example usage
if __name__ == "__main__":
    # Test the quaternion PhysNet
    model = QuaternionPhysNet_padding_Encoder_Decoder_MAX(frames=128)
    
    # Create dummy input
    batch_size = 2
    frames = 128
    height = width = 128
    x = torch.randn(batch_size, 3, frames, height, width)
    
    # Forward pass
    rPPG, x_visual, x_visual3232, x_visual1616 = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output rPPG shape: {rPPG.shape}")
    print(f"Visual features shapes: {x_visual.shape}, {x_visual3232.shape}, {x_visual1616.shape}")