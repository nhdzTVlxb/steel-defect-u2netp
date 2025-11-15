import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1, dilation=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
 
class XceptionABlock(nn.Module):
    """
    Base Block for XceptionA mentioned in DFANet paper.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(XceptionABlock, self).__init__()
        self.conv1 = nn.Sequential(
            depthwise_separable_conv(in_channels, out_channels //4, stride=stride),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            depthwise_separable_conv(out_channels //4, out_channels //4),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            depthwise_separable_conv(out_channels //4, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
 
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
 
    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        identity = self.skip(x)
        
        return residual + identity
        
class enc(nn.Module):
    """
    encoder block
    """
    def __init__(self, in_channels, out_channels, stride=2, num_repeat=3):
        super(enc, self).__init__()
        stacks = [XceptionABlock(in_channels, out_channels, stride=2)]
        for x in range(num_repeat - 1):
            stacks.append(XceptionABlock(out_channels, out_channels))
        self.build = nn.Sequential(*stacks)
        
    def forward(self, x):
        x = self.build(x)
        return x
    
class ChannelAttention(nn.Module):
    """
        channel attention module
    """
    def __init__(self, in_channels, out_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1000, bias=False),
            nn.ReLU(),
            nn.Linear(1000, out_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)
 
class SubBranch(nn.Module):
    """
        create 3 Sub Branches in DFANet
        channel_cfg: the chnnels of each enc stage
        branch_index: the index of each sub branch
    """
    def __init__(self, channel_cfg, branch_index):
        super(SubBranch, self).__init__()
        self.enc2 = enc(channel_cfg[0], 48, num_repeat=3)
        self.enc3 = enc(channel_cfg[1],96,num_repeat=6)
        self.enc4 = enc(channel_cfg[2],192,num_repeat=3)
        self.fc_atten = ChannelAttention(192, 192)
        self.branch_index = branch_index
            
        
    def forward(self,x0,*args):
        out0=self.enc2(x0)
        if self.branch_index in [1,2]:
            out1 = self.enc3(torch.cat([out0,args[0]],1))
            out2 = self.enc4(torch.cat([out1,args[1]],1))
        else:
            out1 = self.enc3(out0)
            out2 = self.enc4(out1)
        out3 = self.fc_atten(out2)
        return [out0, out1, out2, out3]
    
class XceptionA(nn.Module):
    """
    channel_cfg: the all channels in each enc blocks
    """
    def __init__(self, channel_cfg, num_classes=33):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        self.branch = SubBranch(channel_cfg, branch_index=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(192, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )
    
    def forward(self, x):
        b,c,_,_ = x.szie()
        x = self.conv1(x)
        _,_,_,x = self.branch(x)
        x = self.avg_pool(x).view(b,-1)
        x = self.classifier(x)
        
        return x
 
class DFA_Encoder(nn.Module):
    def __init__(self, channel_cfg):
        super(DFA_Encoder, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )
        self.branch0=SubBranch(channel_cfg[0],branch_index=0)
        self.branch1=SubBranch(channel_cfg[1],branch_index=1)
        self.branch2=SubBranch(channel_cfg[2],branch_index=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x0,x1,x2,x5=self.branch0(x)
        x3=F.interpolate(x5,x0.size()[2:],mode='bilinear',align_corners=True)
        
        x1,x2,x3,x6=self.branch1(torch.cat([x0,x3],1),x1,x2)
        x4=F.interpolate(x6,x1.size()[2:],mode='bilinear',align_corners=True)
        
        x2,x3,x4,x7=self.branch2(torch.cat([x1,x4],1),x2,x3)
 
        return [x0,x1,x2,x5,x6,x7]
    
class DFA_Decoder(nn.Module):
    """
        the DFA decoder 
    """
    def __init__(self, decode_channels, num_classes):
        super(DFA_Decoder, self).__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv_add = nn.Sequential(
            nn.Conv2d(in_channels=decode_channels, out_channels=decode_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv_cls = nn.Conv2d(in_channels=decode_channels, out_channels=num_classes, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x0, x1, x2, x3, x4, x5):
        x0 = self.conv0(x0)
        x1 = F.interpolate(self.conv1(x1),x0.size()[2:],mode='bilinear',align_corners=True)
        x2 = F.interpolate(self.conv2(x2),x0.size()[2:],mode='bilinear',align_corners=True)
        x3 = F.interpolate(self.conv3(x3),x0.size()[2:],mode='bilinear',align_corners=True)
        x4 = F.interpolate(self.conv5(x4),x0.size()[2:],mode='bilinear',align_corners=True)
        x5 = F.interpolate(self.conv5(x5),x0.size()[2:],mode='bilinear',align_corners=True)
        
        x_shallow = self.conv_add(x0+x1+x2)
        
        x = self.conv_cls(x_shallow+x3+x4+x5)
        x=F.interpolate(x,scale_factor=4,mode='bilinear',align_corners=True)
        return x
    
class DFANet(nn.Module):
    def __init__(self,decoder_channel=32,num_classes=4):
        super(DFANet,self).__init__()
            #可以用以下代码进行测试
        channel_cfg=[[8,48,96],
            [240,144,288],
            [240,144,288]]
        self.encoder=DFA_Encoder(channel_cfg)
        self.decoder=DFA_Decoder(decoder_channel,num_classes)
 
    def forward(self,x):
        x0,x1,x2,x3,x4,x5=self.encoder(x)
        x=self.decoder(x0,x1,x2,x3,x4,x5)
        return x
    
if __name__=="__main__":
    #可以用以下代码进行测试
    model = DFANet().cuda()
    print(summary(model,input_size=(3,200,200)))
    model = model.cuda()
    a = torch.ones([16, 3, 224, 224])
    a = a.cuda()
    out = model(a)
    print(out.shape)