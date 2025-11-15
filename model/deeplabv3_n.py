import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torch.utils.model_zoo as model_zoo
from functools import partial

BatchNorm2d = nn.BatchNorm2d

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # Pointwise convolution
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 256, 256, 32 -> 256, 256, 16
            [6, 24, 2, 2],  # 256, 256, 16 -> 128, 128, 24
            [6, 32, 3, 2],  # 128, 128, 24 -> 64, 64, 32
            [6, 64, 4, 2],  # 64, 64, 32 -> 32, 32, 64
            [6, 96, 3, 1],  # 32, 32, 64 -> 32, 32, 96
            [6, 160, 3, 2], # 32, 32, 96 -> 16, 16, 160
            [6, 320, 1, 1], # 16, 16, 160 -> 16, 16, 320
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # Initial convolution
        self.features = [conv_bn(3, input_channel, 2)]

        # Building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # Last convolution
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)

def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'), strict=False)
    return model

class MobileNetV21(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV21, self).__init__()
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 

import torch
import torch.nn as nn
from functools import partial
from torchvision.models import convnext_large,convnext_tiny, ConvNeXt_Large_Weights,ConvNeXt_Tiny_Weights

class ConvNeXtBackbone(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(ConvNeXtBackbone, self).__init__()
        
        # 初始化 ConvNeXt 模型
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            model = convnext_tiny(weights=weights)
        else:
            model = convnext_tiny(weights=None)
        
        # 移除分类头
        self.features = nn.Sequential(*list(model.features.children()))
        
        self.total_idx = len(self.features)
        # 下采样的层索引；根据 ConvNeXt 的具体架构可能需要调整
        self.down_idx = [2, 4, 6, 8]  # 示例索引，可能需要根据实际情况调整
        
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        else:
            raise ValueError("downsample_factor 必须为 8 或 16")
    
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if hasattr(m, 'stride') and m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if hasattr(m, 'kernel_size') and m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
    
    def forward(self, x):
        low_level_features = []
        # 定义哪些层对应于低级特征
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in [2, 4, 6, 8]:  # 示例索引，根据实际架构调整
                low_level_features.append(x)
        
        # 这里返回第一个低级特征和最终的高级特征
        return low_level_features[0], x

# 示例用法：




# ASPP Module
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),    
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),    
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),    
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),        
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        # Five branches
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        
        # Global average pooling branch
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        
        # Concatenate all branches
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query    = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C'
        proj_key      = self.key_conv(x).view(batch_size, -1, width * height)                      # B x C' x N
        energy        = torch.bmm(proj_query, proj_key)                                            # B x N x N
        attention     = self.softmax(energy)                                                       # B x N x N
        proj_value    = self.value_conv(x).view(batch_size, -1, width * height)                   # B x C x N
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                                 # B x C x N
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

# Modified DeepLabv3 with Self-Attention
class self_net(nn.Module):
    def __init__(self, num_classes=4, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(self_net, self).__init__()
        from functools import partial
        
        if backbone == "mobilenet":
            # Initialize MobileNetV2 backbone
            self.backbone = MobileNetV21(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet.'.format(backbone))
        
        # ASPP module
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        # Self-Attention module
        self.self_attention = SelfAttention(in_channels=256)
        
        # Shortcut convolution for low-level features
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Combined convolution after concatenation
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),  # Combined channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.1),
        )
        
        # Classification layer
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        
        # Backbone forward pass
        low_level_features, x = self.backbone(x)
        
        # Apply ASPP
        x = self.aspp(x)
        
        # Apply Self-Attention
        x = self.self_attention(x)
        
        # Process low-level features
        low_level_features = self.shortcut_conv(low_level_features)
        print(low_level_features.shape,x.shape)
        # Upsample ASPP output to match low-level features
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        print(x.shape)
        # Concatenate and apply convolution
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        
        # Classification
        x = self.cls_conv(x)
        
        # Upsample to original image size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
if __name__ == '__main__':
    backbone = ConvNeXtBackbone(downsample_factor=8, pretrained=True)
    input_tensor = torch.randn(1, 3, 224, 224)
    low_level, high_level = backbone(input_tensor)
    print(low_level.shape, high_level.shape)
    ras = self_net().cuda()
    input_tensor = torch.randn(2, 3, 200, 200).cuda()

    out = ras(input_tensor)
    print(out.shape)  