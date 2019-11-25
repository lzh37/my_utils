import torch
import torch.nn as nn
import numpy as np


dtype = torch.float32

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


 
class ResBlock(nn.Module):
    # input --> bn --> relu --> conv --> bn --> relu --> conv --> res
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        bn_1 = nn.BatchNorm2d(input_channel)
        relu = nn.ReLU(inplace=True)
        conv_1 = nn.Conv2d(input_channel,out_channel,(3,3),padding=1)
        bn_2 = nn.BatchNorm2d(out_channel)
        conv_2 = nn.Conv2d(out_channel,out_channel,(3,3),padding=1)
        if self.input_channel != self.out_channel:
            self.conv = nn.Conv2d(input_channel,out_channel,(1,1)).to(device=device, dtype=dtype)
        self.block = nn.Sequential(bn_1, relu, conv_1, bn_2, relu, conv_2).to(device=device, dtype=dtype)
        
    def forward(self, x):
        if self.input_channel == self.out_channel:
            x = x + self.block(x)
        else : 
            x = self.block(x) + self.conv(x)
        return x
        
        
class TwoResBlock(nn.Module):
    # ResBlock --> ResBlock
    def __init__(self, input_channel, out_channel):
        super().__init__()
        res_1 = ResBlock(input_channel, out_channel)
        res_2 = ResBlock(out_channel, out_channel)
        res_3 = ResBlock(out_channel, out_channel)
        self.block = nn.Sequential(res_1, res_2, res_3)
        self.res_output = None
        
    def forward(self, x):
        self.res_output = self.block(x)
        return self.res_output
    
        
        
class EncoderBlock(nn.Module):
    # TwoResBlock --> Maxpooling --> Dropout
    def __init__(self,input_channel,out_channel,dropout=0.1):
        super().__init__()
        self.two_res_block = TwoResBlock(input_channel,out_channel)
        max_pool = nn.MaxPool2d(2,stride=2)
        drop_out = nn.Dropout2d(p=dropout)
        self.block = nn.Sequential(self.two_res_block, max_pool, drop_out)
        self.res_output = None
        
    def forward(self, x):
        x = self.block(x)
        self.res_output = self.two_res_block.res_output
        return x
        
        
class Encoder(nn.Module):
    # EncoderBlock * 4 + TwoResBlock
    # channels 需加上输入图像的通道  [3,16,32,64,128,256]
    def __init__(self,channels):
        super().__init__()
        self.layers = []
        for i in range(4):
            self.layers.append(EncoderBlock(channels[i],channels[i+1]))
        self.layers.append(TwoResBlock(channels[4],channels[5]))
        self.res_outputs = []
        self.encoder = nn.Sequential(*self.layers)
        
        
    def forward(self,x):
        x = self.encoder(x)
        self.res_outputs = [self.layers[i].res_output for i in range(5)]
        return x, self.res_outputs
        
        
        
class DecoderBlock(nn.Module):
    # ConvTranspose2d --> cat --> Dropout --> TwoResBlock
    def __init__(self,in_channel,out_channel,dropout=0.1):
        super().__init__()
        convt = nn.ConvTranspose2d(in_channel,out_channel,(3,3),stride=(2,2),padding=1,output_padding=1).to(device=device, dtype=dtype)
        dropout = nn.Dropout2d(p=dropout)
        two_res_block = TwoResBlock(in_channel,out_channel)
        self.layers = [convt, dropout, two_res_block]
        
        
    def forward(self, x):
        x, encoder_res_output = x
        x = self.layers[0](x)
        x = torch.cat([x, encoder_res_output], dim=1)
        x = self.layers[1](x)
        x = self.layers[2](x)
        return x
        
        
class Decoder(nn.Module):
    # DecoderBlock * 4 --> Conv2d
    def __init__(self,channels):  # [3,16,32,64,128,256]
        super().__init__()
        self.layers = []
        for i in range(4):
            dec = DecoderBlock(channels[5-i],channels[4-i])
            self.layers.append(dec)
        conv = nn.Conv2d(channels[1],1,(1,1)).to(device=device, dtype=dtype)
        sig = nn.Sigmoid().to(device=device, dtype=dtype)
        self.layers.append(conv)
        self.layers.append(sig)
        
        
    def forward(self,x):
        x, encoder_res_outputs = x
        for i in range(4):
            x = self.layers[i]((x,encoder_res_outputs[3-i]))
        x = self.layers[4](x)
        output = self.layers[5](x)
        return output
        
        
class ResUnet(nn.Module):
    def __init__(self,channels):
        super().__init__()
        encoder = Encoder(channels)
        decoder = Decoder(channels)
        self.net = nn.Sequential(encoder,decoder)
    def forward(self,x):
        return self.net(x)
        