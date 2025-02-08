import torch
from torch import nn
from utils.hparameter import payload_length


class unet_decoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=8, depth=4, upsample_factor=2, kernel_size=3):
        super(unet_decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.upsample_factor = upsample_factor
        self.kernel_size = kernel_size
    
        self.first_conv = nn.Sequential(
                        nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2, padding_mode='reflect'),
                        nn.BatchNorm2d(base_channels),
                        nn.LeakyReLU(0.1, inplace=True))

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.ModuleList()

        encoder_in_channels_list = [in_channels] + [(2**i) * self.base_channels for i in range(self.depth-1)]
        encoder_out_channels_list = [(2**i) * self.base_channels for i in range(self.depth)]
        
        # Build encoder
        for i in range(depth):
            self.encoder.append(self.build_encoder_block(encoder_in_channels_list[i], encoder_out_channels_list[i]))
            self.pool.append(nn.Conv2d(encoder_out_channels_list[i], encoder_out_channels_list[i],
                                       kernel_size=(4,3), stride=(2,1), padding=1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        
        self.conv_out = nn.Sequential(
            nn.Conv2d(encoder_out_channels_list[-1], 1, 3, 1, 1),
        )
        self.sig = nn.Sigmoid()
        self.fc = nn.Sequential(nn.Linear(in_features=39, out_features=1))
    
    def build_encoder_block(self, in_channels, out_channels, kernel_size=15):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),       
        )
        return block

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        N, _, _, _ = x.size()
        chunks = []
        group_length = x.size(-1) // payload_length
        remainder = x.size(-1) % payload_length

        x = x[:,:,:, :-remainder]
        for i in range(self.depth):
            x = self.encoder[i](x)
            x = self.pool[i](x)
        
        x = self.conv_out(x)

        for i in range(payload_length):
            chunk = x[:,:,:,int(i*group_length):int((i+1)*group_length)]
            chunks.append(chunk)

        x = torch.cat(chunks, dim=0)

        max_values_row, max_indices_row = torch.max(torch.abs(x), dim=2, keepdim=True)
        max_values_col, max_indices_col = torch.max(max_values_row, dim=3, keepdim=True)
        x = torch.gather(x, dim=2, index=max_indices_row)
        x = torch.gather(x, dim=3, index=max_indices_col)
        
        chunks = []
        for i in range(payload_length):
            chunk = x[int(i*N):int((i+1)*N),:,:,:]
            chunks.append(chunk)
        x = torch.cat(chunks, dim=-1).squeeze(1)

        return x