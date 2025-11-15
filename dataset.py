import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class SemanticSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir (str): 图像文件夹路径。
            masks_dir (str): 掩码文件夹路径。
            transform (callable, optional): 可选的变换函数。
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # 初始化图像和掩码的文件名列表
        self.images = sorted([file for file in os.listdir(images_dir) 
                              if os.path.isfile(os.path.join(images_dir, file))])
        self.masks = sorted([file for file in os.listdir(masks_dir) 
                             if os.path.isfile(os.path.join(masks_dir, file))])
        
        # 确保图像和掩码数量相同
        assert len(self.images) == len(self.masks), "图像和掩码数量不匹配"
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 获取图像和掩码的路径
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        print(img_path,mask_path)
        # 打开图像和掩码
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")  # 假设掩码是灰度图
        
        if self.transform:
            # 应用数据增强变换
            augmented_image = self.transform(image)
            augmented_mask = self.transform(mask)
        else:
            augmented_image = image
            augmented_mask = mask
        sample = {'imidx':idx, 'image':augmented_image, 'label':augmented_mask}
        return sample
if __name__=="__main__": 
    # 创建数据增强变换
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        # 添加其他需要的变换
        transforms.ToTensor(),
        # 如果需要对mask进行tensor转换，可能还需要一个专门的mask_transform
    ])
    val_data_transform = transforms.Compose([
        # 添加其他需要的变换
        transforms.ToTensor(),
        # 如果需要对mask进行tensor转换，可能还需要一个专门的mask_transform
    ])
    # 实例化自定义Dataset
    dataset = SemanticSegmentationDataset(
        images_dir=r'images\training',
        masks_dir=r'annotations\training',
        transform=data_transform
    )
    
    # 使用dataset[0]来测试数据增强是否正确应用
    _,augmented_image, augmented_mask = dataset[2]

    print(augmented_image.shape,augmented_mask.shape)