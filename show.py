# import numpy as np
# import matplotlib.pyplot as plt

# # 加载两个 .npy 文件
# file1 = r'submit\test_ground_truths\000003.npy'
# file2 = r'submit\test_predictions\000003.npy'

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

# # 加载三个 .npy 文件
# label_file = r'submit\test_ground_truths\000001.npy'     # 标签文件
# pred_file1 = r'submit\test_predictions\000001.npy' # 预测文件1
# pred_file2 = r'submit\baseline_predictions\000001.npy'# 预测文件2

# label = np.load(label_file)
# pred1 = np.load(pred_file1)
# pred2 = np.load(pred_file2)

# # 可视化三个文件
# plt.figure(figsize=(15, 5))

# # 标签
# plt.subplot(1, 3, 1)
# plt.imshow(label)
# plt.title('Label')
# plt.colorbar()

# # 预测文件1
# plt.subplot(1, 3, 2)
# plt.imshow(pred1)
# plt.title('Prediction 1')
# plt.colorbar()

# # 预测文件2
# plt.subplot(1, 3, 3)
# plt.imshow(pred2)
# plt.title('Prediction 2')
# plt.colorbar()

# plt.show()

# # 计算IoU函数
# def calculate_iou(pred, label, num_classes=4):
#     ious = []
#     for cls in range(num_classes):
#         pred_cls = (pred == cls)
#         label_cls = (label == cls)
#         intersection = np.logical_and(pred_cls, label_cls).sum()
#         union = np.logical_or(pred_cls, label_cls).sum()
        
#         if union == 0:
#             iou = np.nan  # 如果某类别没有区域
#         else:
#             iou = intersection / union
#         ious.append(iou)
#     return ious

# # 计算每个预测与标签的IoU
# iou_pred1 = calculate_iou(pred1, label)
# iou_pred2 = calculate_iou(pred2, label)

# # 输出IoU结果
# print("IoU for Prediction 1: ", iou_pred1)
# print("IoU for Prediction 2: ", iou_pred2)

# # 计算平均IoU，忽略背景
# miou_pred1 = np.nanmean(iou_pred1[1:])
# miou_pred2 = np.nanmean(iou_pred2[1:])

# print("mIoU (without background) for Prediction 1: ", miou_pred1)
# print("mIoU (without background) for Prediction 2: ", miou_pred2)


import numpy as np
import os

# 文件夹路径
label_folder = r'E:\python\NEU_Seg\NEU_seg1\submit\test_ground_truths'       # 标签文件夹
pred1_folder = r'E:\python\NEU_Seg\NEU_seg1\submit\test_predictions'      # 预测1文件夹
pred2_folder = r'E:\python\NEU_Seg\NEU_seg1\submit\baseline_predictions'     # 预测2文件夹


# 计算IoU函数
def calculate_iou(pred, label, num_classes=4):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        label_cls = (label == cls)
        intersection = np.logical_and(pred_cls, label_cls).sum()
        union = np.logical_or(pred_cls, label_cls).sum()
        
        if union == 0:
            continue  # 如果某类别没有区域，跳过
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

# 获取所有文件名
label_files = sorted(os.listdir(label_folder))
pred1_files = sorted(os.listdir(pred1_folder))
pred2_files = sorted(os.listdir(pred2_folder))

# 确保每个文件夹中的文件数量一致
assert len(label_files) == len(pred1_files) == len(pred2_files), "文件数量不一致"

# 初始化 IoU 累加器和有效类别计数器
iou_sum_pred1 = np.zeros(4)
iou_sum_pred2 = np.zeros(4)
class_count_pred1 = np.zeros(4)
class_count_pred2 = np.zeros(4)
num_files = len(label_files)

# 遍历每个文件，计算每类 IoU
for i in range(num_files):
    label = np.load(os.path.join(label_folder, label_files[i]))
    pred1 = np.load(os.path.join(pred1_folder, pred1_files[i]))
    pred2 = np.load(os.path.join(pred2_folder, pred2_files[i]))
    
    # 计算两个预测文件的 IoU
    iou_pred1 = calculate_iou(pred1, label)
    iou_pred2 = calculate_iou(pred2, label)
    
    # 累加每类的 IoU，并计数非空类别
    for cls, iou in enumerate(iou_pred1):
        if not np.isnan(iou):  # 只累加有效IoU
            iou_sum_pred1[cls] += iou
            class_count_pred1[cls] += 1
    
    for cls, iou in enumerate(iou_pred2):
        if not np.isnan(iou):  # 只累加有效IoU
            iou_sum_pred2[cls] += iou
            class_count_pred2[cls] += 1

# 计算每类的平均 IoU
mean_iou_pred1 = np.divide(iou_sum_pred1, class_count_pred1, where=class_count_pred1!=0)
mean_iou_pred2 = np.divide(iou_sum_pred2, class_count_pred2, where=class_count_pred2!=0)

# 输出每类平均IoU
for cls in range(4):
    if class_count_pred1[cls] > 0:
        print(f"Class {cls} - Prediction 1 Average IoU: {mean_iou_pred1[cls]}")
    else:
        print(f"Class {cls} - Prediction 1: No valid IoU for this class")
    
    if class_count_pred2[cls] > 0:
        print(f"Class {cls} - Prediction 2 Average IoU: {mean_iou_pred2[cls]}")
    else:
        print(f"Class {cls} - Prediction 2: No valid IoU for this class")

# 计算不包含背景的平均 IoU (mIoU)
miou_pred1 = np.nanmean(mean_iou_pred1[1:]) if np.any(class_count_pred1[1:]) else 0
miou_pred2 = np.nanmean(mean_iou_pred2[1:]) if np.any(class_count_pred2[1:]) else 0

print(f"Prediction 1 mIoU (without background): {miou_pred1}")
print(f"Prediction 2 mIoU (without background): {miou_pred2}")
