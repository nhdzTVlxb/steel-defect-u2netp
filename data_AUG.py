# 导入数据增强工具
import Augmentor

# 确定原始图像存储路径以及掩码mask文件存储路径
p = Augmentor.Pipeline(r"images_png")
p.ground_truth(r"annotations/training/")

# 图像旋转：按照概率0.8执行，最大左旋角度10，最大右旋角度10
# rotate操作默认在对原图像进行旋转之后进行裁剪，输出与原图像同样大小的增强图像
p.rotate(probability=0.8, max_left_rotation=10, max_right_rotation=10)

# 图像上下镜像： 按照概率0.5执行
p.flip_top_bottom(probability=0.5)

# 图像左右镜像： 按照概率0.5执行
p.flip_left_right(probability=0.5)
p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.2)  
p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.2)   
# 图像等比缩放，按照概率1执行，整个图片放大，像素变多
# p.scale(probability=1, scale_factor=1.3)

# 图像放大：放大后像素不变，先根据percentage_area放大，后按照原图像素大小进行裁剪
# 按照概率0.4执行，面积为原始图0.9倍
# p.zoom_random(probability=0.4, percentage_area=0.9)
# p.scale(probability=1,scale_factor=1.25)
# #缩小
# p.zoom_random(probability=1,percentage_area=0.4)
# #从中心裁剪
# p.crop_centre(probability=1,percentage_area=0.6)
# #按大小裁剪
# p.crop_by_size(probability=1,width=100,height=100)
#垂直形变
# p.skew_tilt(probability=1,magnitude=1)
# #斜四角形变
# p.skew_corner(probability=1,magnitude=1)
# # #弹性扭曲
# # p.random_distortion(probability=1,grid_height=5,grid_width=16,magnitude=8)
# #错切变换
# p.shear(probability=1,max_shear_left=25,max_shear_right=25)

# 最终扩充的数据样本数
p.sample(2000)