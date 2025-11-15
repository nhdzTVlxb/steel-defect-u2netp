import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from albumentations import *
import numpy as np
import glob
import os

from get_acc import *
from data_loader import *

from tqdm import tqdm
from dataset import *

def compute_metrics2(pred, target, num_classes=4):
    # 计算每个类别的IoU
    pred = torch.argmax(pred, dim=1)
    target = target.long().squeeze()
    metric = SegmentationMetric(num_classes) # 3表示有3个分类，有几个分类就填几
    pred=pred.detach().cpu().numpy()
    target=target.detach().cpu().numpy() 
    metric.addBatch(pred, target)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    return pa,cpa,mpa,mIoU,mIoU



epoch_num = 500
batch_size_train = 16
batch_size_val = 1
train_num = 0
val_num = 0



# albumentations_transform = albu.Compose([
#     albu.HorizontalFlip(),
#     albu.OneOf([
#         # albu.RandomContrast(),
#         albu.RandomGamma(),
#         # albu.RandomBrightness(),
#         albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)
#         ], p=0.3),
#     albu.OneOf([
#         # albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#         albu.GridDistortion(),
#         albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
#         ], p=0.3),
#     albu.ShiftScaleRotate(),
#     # albu.Resize(img_size,img_size,always_apply=True),
# ])


def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)

albumentations_transform =strong_aug()

def get_image_label_list(image_dir, label_dir, image_ext, label_ext):
    # 获取图像列表
    img_name_list = glob.glob(os.path.join(image_dir, '*' + image_ext))
    
    # 生成对应的标签列表
    lbl_name_list = []
    for img_path in img_name_list:
        img_name = os.path.basename(img_path)
        imidx = os.path.splitext(img_name)[0]  # 去除后缀
        lbl_name_list.append(os.path.join(label_dir, imidx + label_ext))

    return img_name_list, lbl_name_list

def create_dataloader(image_dir, label_dir, image_ext, label_ext, batch_size, shuffle=True):
    img_name_list, lbl_name_list = get_image_label_list(image_dir, label_dir, image_ext, label_ext)
    
    dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=lbl_name_list,
        transform=transforms.Compose([AlbumentationsTransform(albumentations_transform),ToTensorLab(flag=0)])#
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    print(f"---\n{image_dir} images: {len(img_name_list)}\n{label_dir} labels: {len(lbl_name_list)}\n---")
    
    return dataloader

def create_dataloader1(image_dir, label_dir, image_ext, label_ext, batch_size, shuffle=True):
    img_name_list, lbl_name_list = get_image_label_list(image_dir, label_dir, image_ext, label_ext)
    
    dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=lbl_name_list,
        transform=transforms.Compose([AlbumentationsTransform(albumentations_transform),ToTensorLab(flag=0)])#
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    print(f"---\n{image_dir} images: {len(img_name_list)}\n{label_dir} labels: {len(lbl_name_list)}\n---")
    
    return train_dataloader,val_dataloader
# 参数配置
tra_image_dir = r"images/training_gray/"
tra_label_dir = r"annotations/training/"
val_image_dir = r"images/test_gray/"
val_label_dir = r"annotations/test/"
image_ext = ".jpg"
label_ext = ".png"
# batch_size_train = 4

# 创建训练集和验证集的 DataLoader
salobj_dataloader = create_dataloader(tra_image_dir, tra_label_dir, image_ext, label_ext, batch_size_train)
salobj_valdataloader = create_dataloader(val_image_dir, val_label_dir, image_ext, label_ext, batch_size_train)
# salobj_dataloader ,salobj_valdataloader = create_dataloader1(tra_image_dir, tra_label_dir, image_ext, label_ext, batch_size_train)

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=4, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights.cuda(), ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

class IouLoss(nn.Module):
    def __init__(self, num_classes=4, reduction='mean', smooth=1):
        """
        多分类 IOU 损失函数

        参数:
            num_classes (int): 类别数量
            reduction (str): 'mean' | 'sum' | 'none'，决定如何汇总损失
            smooth (float): 平滑因子，防止分母为0
        """
        super(IouLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        前向传播计算 IOU 损失

        参数:
            inputs (Tensor): 模型输出，形状为 (N, C, H, W)
            targets (Tensor): 真实标签，形状为 (N, H, W)
        
        返回:
            Tensor: 计算得到的 IOU 损失
        """
        # 应用 softmax 以获取每个类别的概率
        inputs = F.softmax(inputs, dim=1)  # Shape: (N, C, H, W)

        # 将 targets 转换为 one-hot 编码，形状变为 (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        # 计算每个类别的交集和并集
        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))  # Shape: (C,)
        total = (inputs + targets_one_hot).sum(dim=(0, 2, 3))       # Shape: (C,)
        union = total - intersection

        # 计算每个类别的 IOU
        iou = (intersection + self.smooth) / (union + self.smooth)  # Shape: (C,)

        # 计算 IOU 损失
        iou_loss = 1 - iou  # Shape: (C,)

        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        else:
            return iou_loss  # 返回每个类别的损失
        
class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, reduction='mean', smooth=1):
        """
        多分类 Dice 损失函数

        参数:
            num_classes (int): 类别数量
            reduction (str): 'mean' | 'sum' | 'none'，决定如何汇总损失
            smooth (float): 平滑因子，防止分母为0
        """
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        前向传播计算 Dice 损失

        参数:
            inputs (Tensor): 模型输出，形状为 (N, C, H, W)
            targets (Tensor): 真实标签，形状为 (N, H, W)
        
        返回:
            Tensor: 计算得到的 Dice 损失
        """
        # 应用 softmax 以获取每个类别的概率
        inputs = F.softmax(inputs, dim=1)  # Shape: (N, C, H, W)

        # 将 targets 转换为 one-hot 编码，形状变为 (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        # 计算每个类别的交集与总和
        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))  # Shape: (C,)
        total = (inputs + targets_one_hot).sum(dim=(0, 2, 3))  # Shape: (C,)
        
        # 计算每个类别的 Dice 系数
        dice = (2 * intersection + self.smooth) / (total + self.smooth)  # Shape: (C,)
        
        # 计算 Dice 损失
        dice_loss = 1 - dice  # Shape: (C,)

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss  # 返回每个类别的损失
        
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2, reduction='mean'):
        """
        多分类 Focal Loss 损失函数

        参数:
            num_classes (int): 类别数量
            alpha (Tensor, optional): 类别权重，形状为 (C,)，若不提供则默认为每个类别权重相等
            gamma (float): Focal Loss 的调节参数，通常取值为 2
            reduction (str): 'mean' | 'sum' | 'none'，决定如何汇总损失
        """
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.alpha is not None:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        """
        前向传播计算 Focal Loss

        参数:
            inputs (Tensor): 模型输出，形状为 (N, C, H, W)
            targets (Tensor): 真实标签，形状为 (N, H, W)
        
        返回:
            Tensor: 计算得到的 Focal Loss
        """
        # 对 inputs 应用 softmax，获得类别概率
        inputs = F.softmax(inputs, dim=1)  # Shape: (N, C, H, W)

        # 将 targets 转换为 one-hot 编码
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # Shape: (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # Shape: (N, C, H, W)

        # 计算交叉熵损失的基础项
        BCE_loss = - (targets_one_hot * torch.log(inputs + 1e-8))  # Shape: (N, C, H, W)

        # 计算Focal Loss的调节项
        pt = inputs * targets_one_hot + (1 - inputs) * (1 - targets_one_hot)  # Shape: (N, C, H, W)
        focal_weight = (1 - pt) ** self.gamma  # Shape: (N, C, H, W)

        # 如果提供了 alpha 权重，对每个类别应用权重
        if self.alpha is not None:
            alpha_factor = self.alpha[None, :, None, None]  # Shape: (1, C, 1, 1)
            focal_weight = focal_weight * alpha_factor

        # 计算最终的 Focal Loss
        loss = focal_weight * BCE_loss  # Shape: (N, C, H, W)

        # 根据 reduction 参数进行汇总
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 返回每个像素的损失

class BoundaryLoss(nn.Module):
    def __init__(self, num_classes=4, reduction='mean'):
        """
        多分类分割边界损失函数

        参数:
            num_classes (int): 类别数量
            reduction (str): 'mean' | 'sum' | 'none'，决定如何汇总损失
        """
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

        # 定义单通道的 Sobel 卷积核
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [ 0,  0,  0],
                                       [ 1,  2,  1]], dtype=torch.float32)

        # 扩展卷积核以匹配类别数
        sobel_kernel_x = sobel_kernel_x.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        sobel_kernel_y = sobel_kernel_y.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        # 复制卷积核 num_classes 次
        self.sobel_kernel_x = sobel_kernel_x.repeat(self.num_classes, 1, 1, 1).cuda()  # [C, 1, 3, 3]
        self.sobel_kernel_y = sobel_kernel_y.repeat(self.num_classes, 1, 1, 1).cuda()   # [C, 1, 3, 3]

        # 将卷积核注册为 buffer，以便在使用 GPU 时自动转移
        self.register_buffer('sobel_kernel_x_buffer', self.sobel_kernel_x)
        self.register_buffer('sobel_kernel_y_buffer', self.sobel_kernel_y)

    def forward(self, inputs, targets):
        """
        前向传播计算边界损失

        参数:
            inputs (Tensor): 模型输出，形状为 (N, C, H, W)
            targets (Tensor): 真实标签，形状为 (N, H, W)

        返回:
            Tensor: 计算得到的边界损失
        """
        # 1. 对inputs应用softmax，获得类别概率
        inputs = F.softmax(inputs, dim=1)  # Shape: (N, C, H, W)

        # 2. 将targets转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # Shape: (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # Shape: (N, C, H, W)

        # 3. 计算输入和目标边界梯度
        input_grad = self._compute_gradient(inputs)     # 计算输入的边界梯度
        target_grad = self._compute_gradient(targets_one_hot)  # 计算目标的边界梯度

        # 4. 计算输入与目标边界梯度的差异，作为边界损失
        boundary_loss = F.mse_loss(input_grad, target_grad, reduction='none')  # Shape: (N, C, H, W)

        # 5. 汇总损失
        if self.reduction == 'mean':
            return boundary_loss.mean()
        elif self.reduction == 'sum':
            return boundary_loss.sum()
        else:
            return boundary_loss  # 返回每个像素的损失

    def _compute_gradient(self, tensor):
        """
        计算输入张量的边界梯度

        参数:
            tensor (Tensor): 输入的张量，形状为 (N, C, H, W)

        返回:
            Tensor: 计算得到的梯度幅值图像，形状为 (N, C, H, W)
        """
        # 使用分组卷积对每个类别的通道单独应用Sobel卷积核
        grad_x = F.conv2d(tensor, self.sobel_kernel_x_buffer, padding=1, groups=self.num_classes)
        grad_y = F.conv2d(tensor, self.sobel_kernel_y_buffer, padding=1, groups=self.num_classes)

        # 计算梯度幅值
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # 添加1e-8避免除零错误

        return grad

import numpy as np
import kornia
import torch
import random
import torch.nn as nn

def colorJitter(colorJitter, img_mean, data = None, target = None, s=0.25):
    # s is the strength of colorjitter
    #colorJitter
    if not (data is None):
        if data.shape[1]==3:
            if colorJitter > 0.2:
                img_mean, _ = torch.broadcast_tensors(img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3), data)
                seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s,contrast=s,saturation=s,hue=s))
                data = (data+img_mean)/255
                data = seq(data)
                data = (data*255-img_mean).float()
    return data, target

def gaussian_blur(blur, data = None, target = None):
    if not (data is None):
        if data.shape[1]==3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15,1.15)
                kernel_size_y = int(np.floor(np.ceil(0.1 * data.shape[2]) - 0.5 + np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(np.floor(np.ceil(0.1 * data.shape[3]) - 0.5 + np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target

def flip(flip, data = None, target = None):
    #Flip
    if flip == 1:
        if not (data is None): data = torch.flip(data,(3,))#np.array([np.fliplr(data[i]).copy() for i in range(np.shape(data)[0])])
        if not (target is None):
            target = torch.flip(target,(2,))#np.array([np.fliplr(target[i]).copy() for i in range(np.shape(target)[0])])
    return data, target

def cowMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask, data = torch.broadcast_tensors(mask, data)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        data = (stackedMask*torch.cat((data[::2],data[::2]))+(1-stackedMask)*torch.cat((data[1::2],data[1::2]))).float()
    if not (target is None):
        stackedMask, target = torch.broadcast_tensors(mask, target)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        target = (stackedMask*torch.cat((target[::2],target[::2]))+(1-stackedMask)*torch.cat((target[1::2],target[1::2]))).float()
    return data, target

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target

def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target


def normalize(MEAN, STD, data = None, target = None):
    #Normalize
    if not (data is None):
        if data.shape[1]==3:
            STD = torch.Tensor(STD).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            MEAN = torch.Tensor(MEAN).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            STD, data = torch.broadcast_tensors(STD, data)
            MEAN, data = torch.broadcast_tensors(MEAN, data)
            data = ((data-MEAN)/STD).float()
    return data, target