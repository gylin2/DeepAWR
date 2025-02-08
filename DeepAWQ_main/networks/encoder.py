import torch
from torch import nn
from utils.hparameter import payload_length
import torch.nn.functional as F

class Waveunet(nn.Module):
    def __init__(self, data_depth, in_channels=1, out_channels=1, base_channels=8, depth=4, upsample_factor=2, kernel_size=3):
        super(Waveunet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.upsample_factor = upsample_factor
        self.kernel_size = kernel_size
        self.data_depth = data_depth
        self.conv_expand = nn.Sequential(self._conv1d(1, self.data_depth))
        self.conv_expand2 = nn.Sequential(self._conv2d(1, self.data_depth))
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.up = nn.ModuleList()

        encoder_in_channels_list = [in_channels] + [(2**i) * self.base_channels for i in range(self.depth-1)]
        encoder_out_channels_list = [(2**i) * self.base_channels for i in range(self.depth)]
        
        # Build encoder
        for i in range(depth):
            self.encoder.append(self.build_encoder_block(encoder_in_channels_list[i], encoder_out_channels_list[i]))
            self.pool.append(nn.Conv2d(encoder_out_channels_list[i], encoder_out_channels_list[i],
                                       kernel_size=(4,3), stride=(2,1), padding=1))
        
        self.middle = nn.Sequential(
            nn.Conv2d(encoder_out_channels_list[-1]+self.data_depth, encoder_out_channels_list[-1], 3, stride=1, padding=1),
            nn.BatchNorm2d(encoder_out_channels_list[-1]),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        
        self.fc = nn.Sequential(
            nn.Linear(1, 75),
            )

        decoder_in_channels_list = [(2**i) * self.base_channels for i in range(1, self.depth)] + [
            2 ** self.depth * self.base_channels]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_in_channels_list[::-1]
        
        # Build decoder
        for i in range(depth):
            self.decoder.append(self.build_decoder_block(decoder_in_channels_list[i], decoder_out_channels_list[i]))
            self.up.append(nn.ConvTranspose2d(int(decoder_in_channels_list[i]/2), int(decoder_in_channels_list[i]/2), 
                                              kernel_size=(2,3), stride=(2,1), padding=(0,1)))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels+in_channels, self.out_channels, kernel_size=1),
        )

    def _conv1d(self, in_channels, out_channels):
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    def build_encoder_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),       
        )
        return block
    def build_decoder_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        return block
    def resize_label(A, B):
        size_B = B.size()[-1]
        resized_A = F.interpolate(A, size=size_B, mode='linear', align_corners=True)
        return resized_A

    def forward(self, cover, data, factor=1):
        
        cover = cover.unsqueeze(1)
        group_length = cover.size(-1) // payload_length
        remainder = cover.size(-1) % payload_length
        x = cover[:,:,:, :-remainder]
        
        # Encoder
        skips = []        
        for i in range(self.depth):
            x = self.encoder[i](x)
            skips.append(x)
            x = self.pool[i](x)

        expanded_data = torch.repeat_interleave(data, repeats=group_length, dim=-1)
        data = self.conv_expand(expanded_data).unsqueeze(2).repeat(1,1,x.size()[2],1)


        x = torch.cat([x, data], dim=1)
        x = self.middle(x)
        

        # Decoder
        for i in range(self.depth):
            skip = skips.pop()
            x = self.up[i](x)
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[i](x)

        if remainder > 0:
            x_zeros = torch.zeros_like(x)
            x = torch.cat([x, x_zeros], dim=-1)[:,:,:,:cover.size(-1)]
            
        return (cover + x*factor).squeeze()