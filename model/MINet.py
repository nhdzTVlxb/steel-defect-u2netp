import torch.nn as nn
import torch.nn.functional as F
import torch


class convbnrelu(nn.Module):

    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel), convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu))

    def forward(self, x):
        return self.conv(x)


class ConvOut(nn.Module):

    def __init__(self, in_channel):
        super(ConvOut, self).__init__()
        self.conv = nn.Sequential(nn.Dropout2d(p=0.1), nn.Conv2d(in_channel, 4, 1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class MI_Module(nn.Module):  # Multiscale Interactive Module

    def __init__(self, channel):
        super(MI_Module, self).__init__()
        # Parameters Setting
        self.channel = channel

        # Module layer
        self.Dconv = nn.ModuleList([convbnrelu(channel, channel, k=3, s=1, p=2**i, d=2**i, g=channel) for i in range(4)])

        self.Pconv = nn.ModuleList([convbnrelu(4, 1, k=1, s=1, p=0) for i in range(channel)])

        self.Pfusion = convbnrelu(channel, channel, k=1, s=1, p=0)

        self.relu = nn.ReLU(inplace=True)

    def Shuffle(self, Fea_list, channel):
        MsFea_sp = [torch.chunk(x, channel, dim=1) for x in Fea_list]  # Multi-Scale Feature(Spilit)
        IrFea = [torch.cat((MsFea_sp[0][i], MsFea_sp[1][i], MsFea_sp[2][i], MsFea_sp[3][i]), 1) for i in range(channel)]  # Interactive Feature

        return IrFea

    def forward(self, x):
        MsFea = [conv(x) for conv in self.Dconv]  # Multi-Scale Feature

        IrFea = self.Shuffle(MsFea, self.channel)
        IrFea = [self.Pconv[i](IrFea[i]) for i in range(self.channel)]  # Interactive Feature

        IrFea = [torch.squeeze(x, dim=1) for x in IrFea]
        out = self.Pfusion(torch.stack(IrFea, 1))

        out = self.relu(out + x)

        return out

class self_net(nn.Module):
    def __init__(self):
        super(self_net, self).__init__()
        #MI-based real-time backbone
        ##stage 1
        self.encoder1 = convbnrelu(3, 16, k=3, s=2, p=1)

        ##stage 2
        self.encoder2 = nn.Sequential(
                DSConv3x3(16, 32, stride=2),
                MI_Module(32),
                MI_Module(32),
                MI_Module(32)
                )

        ##stage 3
        self.encoder3 = nn.Sequential(
                DSConv3x3(32, 64, stride=2),
                MI_Module(64),
                MI_Module(64),
                MI_Module(64),
                MI_Module(64)
                )
                
        ##stage 4
        self.encoder4 = nn.Sequential(
                DSConv3x3(64, 96, stride=2),
                MI_Module(96),
                MI_Module(96),
                MI_Module(96),
                MI_Module(96),
                MI_Module(96),
                MI_Module(96)
                )

        ##stage 5
        self.encoder5 = nn.Sequential(
                DSConv3x3(96, 128, stride=2),
                MI_Module(128),
                MI_Module(128),
                MI_Module(128)
                )

        #Decoder
        ##stage 5
        self.decoder5 = nn.Sequential(
                DSConv3x3(128,128, dilation=2),
                DSConv3x3(128,96, dilation=1)
        )

        ##stage 4
        self.decoder4 = nn.Sequential(
                DSConv3x3(96,96, dilation=2),
                DSConv3x3(96,64, dilation=1)
        )

        ##stage 3
        self.decoder3 = nn.Sequential(
                DSConv3x3(64,64, dilation=2),
                DSConv3x3(64,32, dilation=1)
        )

        ##stage 2
        self.decoder2 = nn.Sequential(
                DSConv3x3(32,32, dilation=2),
                DSConv3x3(32,16, dilation=1)
        )

        ##stage 1
        self.decoder1 = nn.Sequential(
                DSConv3x3(16,16, dilation=2),
                DSConv3x3(16,16, dilation=1)
        )

        #Output
        self.conv_out1 = ConvOut(in_channel=16)
        self.conv_out2 = ConvOut(in_channel=16)
        self.conv_out3 = ConvOut(in_channel=32)
        self.conv_out4 = ConvOut(in_channel=64)
        self.conv_out5 = ConvOut(in_channel=96)

    def forward(self, x):
        #MI-based real-time backbone
        ##stage 1
        score1 = self.encoder1(x)
        ##stage 2
        score2 = self.encoder2(score1)
        ##stage 3
        score3 = self.encoder3(score2)
        ##stage 4
        score4 = self.encoder4(score3)
        ##stage 5
        score5 = self.encoder5(score4)

        #Decoder
        ##stage 5
        scored5 = self.decoder5(score5)
        t = interpolate(scored5, score4.size()[2:])
        ##stage 4
        scored4 = self.decoder4(score4 + t)
        t = interpolate(scored4, score3.size()[2:])
        ##stage 3
        scored3 = self.decoder3(score3 + t)
        t = interpolate(scored3, score2.size()[2:])
        ##stage 2
        scored2 = self.decoder2(score2 + t)
        t = interpolate(scored2, score1.size()[2:])
        ##stage 1
        scored1 = self.decoder1(score1 + t)

        #Output
        out1 = self.conv_out1(scored1)
        out2 = self.conv_out2(scored2)
        out3 = self.conv_out3(scored3)
        out4 = self.conv_out4(scored4)
        out5 = self.conv_out5(scored5)

        out1 = interpolate(out1, x.size()[2:])
        out2 = interpolate(out2, x.size()[2:])
        out3 = interpolate(out3, x.size()[2:])
        out4 = interpolate(out4, x.size()[2:])
        out5 = interpolate(out5, x.size()[2:])

        return out1, out2, out3, out4, out5

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)