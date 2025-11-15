import torch
import time
from model.u2net import self_net  # 确保导入你的模型

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_fps(model, input_size, device, num_warmup=100, num_test=1000):
    model.eval()
    model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            dummy_input = torch.randn(1, *input_size).to(device)
            _ = model(dummy_input)
    
    # Measure FPS
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_test):
            dummy_input = torch.randn(1, *input_size).to(device)
            _ = model(dummy_input)
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_test / total_time
    return fps

# 加载模型
model = self_net().cuda()  # 确保模型加载到 GPU 上

# 设置输入尺寸为 1*200*200
input_size = (3, 200, 200)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取模型参数数量
params = count_parameters(model)
print(f"Model Parameters: {params:,}")

# 测量 FPS
fps = measure_fps(model, input_size, device)
print(f"Model FPS: {fps:.2f}")