import torch
import torch.nn as nn
import numpy as np

dtype = torch.float32

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Conv2dReLU(nn.Module):
    # Conv --> (Norm) --> Relu --> Conv --> (Norm) --> Relu
    def __init__(self,input_channel,out_channel,kernel_size=3,use_batchnorm=True):
        super().__init__()
        layers = []
        for i in range(2):
            conv = nn.Conv2d(input_channel,out_channel,(kernel_size,kernel_size),padding=1)
            relu = nn.ReLU(inplace=True)
            layers = np.concatenate((layers,[conv,relu] if not use_batchnorm else [conv,nn.BatchNorm2d(out_channel),relu]))
            input_channel = out_channel
        self.block = nn.Sequential(*layers).to(device=device, dtype=dtype)
        self.conv_output = None   #保留卷积层的输出
    
    def forward(self, x):
        self.conv_output = self.block(x)
        return self.conv_output

       
class EncoderBlock(nn.Module):
    # Conv2dReLU --> Maxpooling --> Dropout
    def __init__(self,input_channel,out_channel,dropout=0.1):
        super().__init__()
        self.conv_block = Conv2dReLU(input_channel,out_channel)
        max_pool = nn.MaxPool2d(2,stride=2)
        drop_out = nn.Dropout2d(p=dropout)
        self.block = nn.Sequential(self.conv_block, max_pool, drop_out)
        self.conv_output = None   #保留卷积层的输出
    
    def forward(self, x):
        x = self.block(x)
        self.conv_output = self.conv_block.conv_output
        return x


class Encoder(nn.Module):
    # EncoderBlock * 4 + Conv2dReLU
    # channels 需加上输入图像的通道  [3,16,32,64,128,256]
    def __init__(self,channels):
        super().__init__()
        self.layers = []
        for i in range(4):
            self.layers.append(EncoderBlock(channels[i],channels[i+1]))
        self.layers.append(Conv2dReLU(channels[4],channels[5]))
        self.conv_outputs = []
        self.encoder = nn.Sequential(*self.layers)
        
    def forward(self,x):
        x = self.encoder(x)
        self.conv_outputs = [self.layers[i].conv_output for i in range(5)]
        return x, self.conv_outputs
        
        
        
class DecoderBlock(nn.Module):
    # ConvTranspose2d --> cat --> Dropout --> Conv2dReLU
    def __init__(self,in_channel,out_channel,kernel_size=3,dropout=0.1):
        super().__init__()
        convt = nn.ConvTranspose2d(in_channel,out_channel,(kernel_size,kernel_size),stride=(2,2),padding=1,output_padding=1).to(device=device, dtype=dtype)
        dropout = nn.Dropout2d(p=dropout)
        conv = Conv2dReLU(in_channel,out_channel)
        self.layers = [convt, dropout, conv]
    
    def forward(self,x):
        x, encoder_conv_output = x
        x = self.layers[0](x)
        x = torch.cat([x,encoder_conv_output],dim=1)
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
        x, encoder_conv_outputs = x
        for i in range(4):
            x = self.layers[i]((x,encoder_conv_outputs[3-i]))
        x = self.layers[4](x)
        output = self.layers[5](x)
        return output
        
          
class Unet(nn.Module):
    def __init__(self,channels):
        super().__init__()
        encoder = Encoder(channels)
        decoder = Decoder(channels)
        self.net = nn.Sequential(encoder,decoder)
    def forward(self,x):
        return self.net(x)
        
