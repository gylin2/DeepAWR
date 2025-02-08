import argparse
import os
import torch
from torchsummary import summary
from utils.hparameter import *
from networks.encoder import Waveunet as Encoder
from networks.decoder import unet_decoder as Decoder
from deepspeed.profiling.flops_profiler import get_model_profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    channels_size = CHANNEL_SIZE
    encoder = Encoder(data_depth)
    decoder = Decoder()

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    summary(encoder, [(160, 1001),(1, 100)])
    kwargs = {'data': torch.randn(1, 1, 100).to(device)}
    flops, macs, params = get_model_profile(encoder, (1,160, 1001,), kwargs=kwargs ,as_string=False)
    print('FLOPs: {}'.format(flops))
    print('Parameters: {}'.format(params))
    print('FLOPs: %.2fG' % (flops / 1e9))
    print('Parameters: %.2fM' % (params / 1e6))

    # summary(decoder, (160, 1001,))
    # flops, macs, params = get_model_profile(decoder, (1,160, 1001,), as_string=False)
    # print('FLOPs: {}'.format(flops))
    # print('Parameters: {}'.format(params))
    # print('FLOPs: %.2fG' % (flops / 1e9))
    # print('Parameters: %.2fM' % (params / 1e6))