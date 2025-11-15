# data loader
from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import albumentations as albu
from albumentations.pytorch import ToTensorV2
#==========================dataset load==========================
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)


		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}


class BrightnessEnhance(object):
    """Enhance brightness of the image in the sample."""

    def __init__(self, factor=1.5):  # factor > 1 to increase brightness, < 1 to decrease
        self.factor = factor

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        # 转换为PIL图像进行亮度增强
        image_pil = Image.fromarray(np.uint8(image * 255))  # 假设image在[0, 1]范围内
        enhancer = ImageEnhance.Brightness(image_pil)
        enhanced_image = enhancer.enhance(self.factor)

        # 转换回ndarray
        enhanced_image = np.array(enhanced_image) / 255.0  # 恢复到[0, 1]范围

        # 归一化和转换为Tensor
        if enhanced_image.ndim == 2:  # 如果增强后的图像是单通道
            enhanced_image = np.expand_dims(enhanced_image, axis=-1)  # 扩展为 (H, W, 1)
        
        tmpImg = np.zeros((enhanced_image.shape[0], enhanced_image.shape[1], 3))

        return {'imidx': torch.from_numpy(imidx), 'image': tmpImg, 'label': label}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'imidx':imidx,'image':image, 'label':label}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)

		image = image/np.max(image)

		if len(image.shape) == 2:  # 只有两个维度，说明是灰度图像
			tmpImg[:,:,0] = (image - 0.485) / 0.229
			tmpImg[:,:,1] = (image - 0.485) / 0.229
			tmpImg[:,:,2] = (image - 0.485) / 0.229
		elif image.shape[2] == 1:  # 处理单通道图像
			tmpImg[:,:,0] = (image[:,:,0] - 0.485) / 0.229
			tmpImg[:,:,1] = (image[:,:,0] - 0.485) / 0.229
			tmpImg[:,:,2] = (image[:,:,0] - 0.485) / 0.229
		else:  # 处理多通道图像
			tmpImg[:,:,0] = (image[:,:,0] - 0.485) / 0.229
			tmpImg[:,:,1] = (image[:,:,1] - 0.456) / 0.224
			tmpImg[:,:,2] = (image[:,:,2] - 0.406) / 0.225


		tmpLbl[:,:,0] = label[:,:,0]


		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		image = image/np.max(image)
		# if image.shape[2]==1:
		# 	tmpImg[:,:,0] = image[:,:,0]
		# 	tmpImg[:,:,1] = image[:,:,0]
		# 	tmpImg[:,:,2] = image[:,:,0]
		# else:
		# 	tmpImg[:,:,0] = image[:,:,0]
		# 	tmpImg[:,:,1] = image[:,:,1]
		# 	tmpImg[:,:,2] = image[:,:,2]
		# print(image)
		if len(image.shape) == 2:  # 只有两个维度，说明是灰度图像
			tmpImg[:,:,0] = (image - 0.485) / 0.229
			tmpImg[:,:,1] = (image - 0.485) / 0.229
			tmpImg[:,:,2] = (image - 0.485) / 0.229
		elif image.shape[2] == 1:  # 处理单通道图像
			tmpImg[:,:,0] = (image[:,:,0] - 0.485) / 0.229
			tmpImg[:,:,1] = (image[:,:,0] - 0.485) / 0.229
			tmpImg[:,:,2] = (image[:,:,0] - 0.485) / 0.229
		else:  # 处理多通道图像
			tmpImg[:,:,0] = (image[:,:,0] - 0.485) / 0.229
			tmpImg[:,:,1] = (image[:,:,1] - 0.456) / 0.224
			tmpImg[:,:,2] = (image[:,:,2] - 0.406) / 0.225

		tmpImg = tmpImg.transpose((2, 0, 1))

		return {'imidx':imidx, 'image':tmpImg, 'label':label}


class AlbumentationsTransform(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # Ensure image is in uint8 format

        # Apply augmentations
        augmented = self.augmentations(image=image, mask=label)
        image = augmented['image']  # Tensor
        label = augmented['mask']   # Still numpy array (if label is a mask)

        # Convert label to tensor (assuming it's a single-channel mask)
        label = torch.from_numpy(label).long()
		
        return {'imidx': imidx, 'image': image, 'label': label}

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):


		image = io.imread(self.image_name_list[idx])
		# print(image.shape)
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label = np.zeros(image.shape)
		else:
			label = io.imread(self.label_name_list[idx])
		# print(image.shape,label.shape,self.image_name_list[idx])	
		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample


print("Finished running data_loader.py")
# import os
# image_ext = '.jpg'
# label_ext = '.png'
# tra_image_dir = r"images\training/"
# tra_label_dir = r"annotations\training/"

# tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)

# tra_lbl_name_list = []
# for img_path in tra_img_name_list:
# 	img_name = img_path.split(os.sep)[-1]

# 	aaa = img_name.split(".")
# 	bbb = aaa[0:-1]
# 	imidx = bbb[0]
# 	for i in range(1,len(bbb)):
# 		imidx = imidx + "." + bbb[i]

# 	tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)

# print("---")
# print("train images: ", len(tra_img_name_list))
# print("train labels: ", len(tra_lbl_name_list))
# print("---")

# train_num = len(tra_img_name_list)

# salobj_dataset = SalObjDataset(
#     img_name_list=tra_img_name_list,
#     lbl_name_list=tra_lbl_name_list,
#     transform=transforms.Compose([
#         BrightnessEnhance(factor=1.5),
#         ToTensorLab(flag=0)]
#         ))
# salobj_dataloader = DataLoader(salobj_dataset, batch_size=1, shuffle=True, num_workers=0)
# result=[]
# for i in range(800):
# 	a=salobj_dataset.__getitem__(i)
# 	print(a['label'].shape)
# 	result.append(a['label'])
# print(np.unique(np.array(result)))

