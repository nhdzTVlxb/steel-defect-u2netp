# savemodel.py
import torch
import os
from model.u2net import U2NETP  # 假设你已经有了 U2NETP 模型的定义

# 假设你的模型已经训练完成
model = U2NETP(3, 1).cuda()

# 确保模型处于评估模式
model.eval()

# 指定保存路径
model_path = 'saved_models/u2netp_AUG_b_l_D_b_s/u2netp_AUG_b_l_D_b_s_best_iou_0.6089_epoch_294.pth'

# 确保保存路径的目录存在
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# 保存模型权重
torch.save(model.state_dict(), model_path)

# 打印保存成功的消息
print(f"Model weights saved to {model_path}")