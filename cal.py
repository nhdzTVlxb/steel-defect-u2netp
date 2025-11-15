import os
import numpy as np
import torch
from model.deeplabv3 import self_net  # 导入您的模型
import time

def calculate_iou(pred, target, num_classes, ignore_classes=[]):
    """
    计算每个类别的IoU，忽略指定的类别。
    
    Args:
        pred (np.ndarray): 预测结果，形状为 [H, W]。
        target (np.ndarray): 标签，形状为 [H, W]。
        num_classes (int): 类别总数。
        ignore_classes (list): 需要忽略的类别列表。
    
    Returns:
        list: 每个类别的IoU（忽略指定类别后）。
    """
    ious = []
    for cls in range(num_classes):
        if cls in ignore_classes:
            continue  # 忽略指定的类别
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # 如果该类别在标签和预测中都不存在，忽略
        else:
            ious.append(intersection / union)
    return ious

def count_parameters(model):
    """
    计算模型的参数数量（以百万为单位）
    
    Args:
        model (torch.nn.Module): PyTorch模型。
    
    Returns:
        float: 参数数量，单位为百万（M）。
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6  # 转换为百万

def load_data(pred_folder, label_folder):
    """
    从预测文件夹和标签文件夹加载对应的.npy文件。
    假设两个文件夹下的.npy文件名完全相同。
    
    Args:
        pred_folder (str): 预测结果文件夹路径。
        label_folder (str): 标签文件夹路径。
    
    Returns:
        tuple: 预测结果和标签的拼接数组。
    """
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.npy')])

    if len(pred_files) == 0:
        raise ValueError(f"预测文件夹 {pred_folder} 中未找到任何 .npy 文件。")
    
    if len(pred_files) != len(label_files):
        raise ValueError("预测文件夹和标签文件夹中的文件数量不一致。")

    # 确保文件名对应
    for pred_file, label_file in zip(pred_files, label_files):
        if pred_file != label_file:
            raise ValueError(f"文件名不匹配: {pred_file} vs {label_file}")

    preds = []
    labels = []
    for file in pred_files:
        pred_path = os.path.join(pred_folder, file)
        label_path = os.path.join(label_folder, file)  # 假设标签文件名与预测文件名相同
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件 {label_path} 不存在。")
        
        pred_data = np.load(pred_path)
        label_data = np.load(label_path)
        
        # 确保数据的维度一致
        if pred_data.shape != label_data.shape:
            raise ValueError(f"预测文件 {file} 和标签文件的维度不一致。")
        
        preds.append(pred_data)
        labels.append(label_data)
    
    # 连接所有样本
    preds = np.concatenate(preds, axis=0)  # 假设第一个维度是样本数
    labels = np.concatenate(labels, axis=0)
    return preds, labels

def main():
    # 设置设备（仅用于计算参数数量，无需GPU）
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # 加载模型以计算参数数量
    model = self_net()
    model.to(device)
    model.eval()

    # 计算参数数量（以百万为单位）
    total_params_m = count_parameters(model)
    print(f"Total Parameters: {total_params_m:.2f}M")

    # 定义预测和标签文件夹路径
    pred_folder = r'submit\baseline_predictions'         # 替换为您的预测文件夹路径
    label_folder = r'submit\test_ground_truths'      # 替换为您的标签文件夹路径

    num_classes = 4  # 类别总数，包括背景类
    ignore_classes = [0]  # 忽略背景类
    classes_to_evaluate = [1, 2, 3]  # 需要计算IoU的类别

    iou_list = []
    total_time = 0
    total_samples = 0

    # 加载所有数据
    print("Loading data...")
    preds, labels = load_data(pred_folder, label_folder)
    print(f"Total samples: {preds.shape[0]}")

    # 计算IoU
    start_time_total = time.time()
    for idx in range(preds.shape[0]):
        pred = preds[idx]
        target = labels[idx]

        # 计算IoU，忽略背景类
        ious = calculate_iou(pred, target, num_classes, ignore_classes=ignore_classes)
        iou_list.append(ious)

        # 计算处理时间（这里只计算数据处理时间，不涉及模型推理）
        # 如果需要更详细的时间统计，可以在此处添加时间记录

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} samples")

    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    total_samples = preds.shape[0]

    # 计算每个类别的平均IoU
    iou_array = np.array(iou_list)  # 形状 [总样本数, num_evaluated_classes]
    mean_ious = np.nanmean(iou_array, axis=0)  # 忽略NaN
    mIoU = np.nanmean(mean_ious)

    # 计算总体FPS（处理速度）
    avg_fps = total_samples / total_time if total_time > 0 else float('inf')

    # 打印结果
    for cls, iou in zip(classes_to_evaluate, mean_ious):
        print(f"Class{cls}_IoU: {iou:.2f}")
    print(f"mIoU: {mIoU:.2f}")
    print(f"FPS: {avg_fps:.2f}")
    print(f"Parameters: {total_params_m:.2f}M")

    # 将结果保存到字典中
    results = {
        "OursModel": {
            "Class1_IoU": float(np.round(mean_ious[0], 2)),
            "Class2_IoU": float(np.round(mean_ious[1], 2)),
            "Class3_IoU": float(np.round(mean_ious[2], 2)),
            "mIoU": float(np.round(mIoU, 2)),
            "FPS": float(np.round(avg_fps, 2)),
            "Parameters": float(np.round(total_params_m, 2))
        }
    }

    print("Results:", results)

if __name__ == "__main__":
    main()
