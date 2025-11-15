# predict.py
import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import random  # 导入 random 模块

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model.u2net import U2NETP  # small version u2net 4.7 MB
from model.u2net import self_net
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 使用示例
set_seed(42)  # 设置随机种子为42

def main():

    # --------- 1. get image path and name ---------
    model_name = 'u2netp'  # u2netp

    image_dir = r'images/test'
    prediction_dir = r'finall/result_' + model_name + '/'
    os.makedirs(prediction_dir, exist_ok=True)
    
    # 更新模型权重文件路径
    model_dir = r'saved_models/u2netp_AUG_b_l_D_b_s/u2netp_AUG_b_l_D_b_s_best_iou_0.6089_epoch_294.pth'
    img_name_list = glob.glob(image_dir + os.sep + '*')

    # --------- 2. dataloader --------- #
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([
            RescaleT(320),  # 确保输入图像大小一致
            ToTensorLab(flag=0)
        ])
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # 加载模型
    net = U2NETP(3, 1).cuda()  # 修改这里
    net.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage.cuda()))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d0, d1, d2, d3, d4, d5, d6 = net(inputs_test)

        # 获取概率值最大的通道
        result = d0.cpu().numpy().squeeze()

        # 将结果保存为 .npy 文件
        np.save(os.path.join(prediction_dir, img_name_list[i_test].split('.')[0].split(os.sep)[-1] + '.npy'), result)

        # 清理内存
        del d0, d1, d2, d3, d4, d5, d6

if __name__ == "__main__":
    main()