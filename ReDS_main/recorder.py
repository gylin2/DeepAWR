"""
This module implements a network using 1D convolutional layers with residual connections.
"""

from torch import nn

# Global configuration flag: if True, use InstanceNorm1d; otherwise, use BatchNorm1d.
Instance_Norm = True

def convblock(in_channels, out_channels, kernel_size=3, padding=1):
    """
    Constructs a 1D convolutional block with normalization and LeakyReLU activation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        padding (int): Padding applied to the convolution. Default is 1.
    
    Returns:
        nn.Sequential: A sequential block containing Conv1d, LeakyReLU, and a normalization layer.
    """

    def conv1d(in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
    
    def forward(in_channels, out_channels, kernel_size=3, padding=1):
        if Instance_Norm:
            return nn.Sequential(
            conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm1d(out_channels, affine=True),
            )
        return nn.Sequential(
            conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
        )

    return forward(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

class Residual_block(nn.Module):
    """
    A residual block that applies two 1D convolution layers with SELU activation and Batch Normalization.

    Args:
        nb_filts (list or tuple): A two-element sequence where:
            - nb_filts[0]: number of input channels.
            - nb_filts[1]: number of output channels.
    """
    def __init__(self, nb_filts):
        super(Residual_block, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(num_features = nb_filts[1])
        
        self.selu = nn.SELU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
            bias=False)
        
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			kernel_size = 3,
            padding = 1,
            bias=False)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 1,
				kernel_size = 3,
				stride = 1,
                bias=False)
            
        else:
            self.downsample = False
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.selu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.selu(out)
        return out
    

class Recorder(nn.Module):
    """
    The Recorder network processes input data through a series of convolutional blocks with residual connections.
    
    Args:
        hidden_size (int): The number of hidden channels. Default is 32.
        channels_size (int): The number of input channels. Default is 1.
    """
    def __init__(self, hidden_size=32, channels_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.channels_size = channels_size
        self._build_models()
        # self.name = self._name()

    # def _name(self):
    #     return "Recorder"

    def _conv1d(self, in_channels, out_channels):
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        self.conv1 = convblock(self.channels_size, self.hidden_size)
        self.conv2 = convblock(self.hidden_size, self.hidden_size)
        self.conv3 = convblock(self.hidden_size, self.hidden_size)
        self.conv4 = convblock(self.hidden_size, self.hidden_size)
        self.conv5 = convblock(self.hidden_size, self.hidden_size)
        self.conv6 = convblock(self.hidden_size, self.hidden_size)
        self.conv7 = convblock(self.hidden_size, self.hidden_size)
        self.conv8 = convblock(self.hidden_size, self.hidden_size)
        self.conv9 = convblock(self.hidden_size, self.hidden_size)
        self.conv10 = convblock(self.hidden_size, self.hidden_size)
        self.conv11 = convblock(self.hidden_size, self.hidden_size)

        self.conv_out = nn.Sequential(
            self._conv1d(self.hidden_size, 1),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(3, self.hidden_size*2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.hidden_size*2, self.hidden_size),
        #     nn.Sigmoid())

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x_list = x
        x_1 = self.conv2(x)
        x_list = x_list + x_1
        x_2 = self.conv3(x_list)
        x_list = x_list + x_2
        x_3 = self.conv4(x_list)
        x_list = x_list + x_3
        x_4 = self.conv5(x_list)
        x_list = x_list + x_4
        x_5 = self.conv6(x_list)
        x_list = x_list + x_5
        x_6 = self.conv7(x_list)
        x_list = x_list + x_6
        x_7 = self.conv8(x_list)
        x_list = x_list + x_7
        x_8 = self.conv9(x_list)
        x_list = x_list + x_8
        x_9 = self.conv10(x_list)
        x_list = x_list + x_9
        x_10 = self.conv11(x_list)
        
        x_out = self.conv_out(x_10)
        return x_out.squeeze(1)
