import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
from data_loader import *
import numpy as np
from PIL import Image
import glob

from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model.deeplabv3 import self_net

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
    image_dir = r'test_B/Img'
    prediction_dir = r'submit/test_predictions/'
    os.makedirs(prediction_dir,exist_ok=True)
    model_dir=r'submit/model.pth'
    img_name_list = glob.glob(image_dir + os.sep + '*')

    # --------- 2. dataloader --------- #
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([
                                            ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    net= self_net().cuda()
    # net= torch.load(model_dir)
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1 = net(inputs_test)

        # 获取概率值最大的通道
        max_channel = torch.argmax(d1, dim=1)  # 获取最大概率的通道
        result=max_channel.cpu().numpy()
        # print(np.unique(result))
        # break
        np.save(prediction_dir+img_name_list[i_test].split('.')[0].split('\\')[-1]+'.npy',result.squeeze())
        del d1



if __name__ == "__main__":
    main()
