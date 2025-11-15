import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import torchvision
import numpy as np
import glob
import os
import shutil
from get_acc import *
from data_loader import *
# from dataset import *
# from model.model import self_net
# from model.deeplabv3 import self_net
# from model.SAMNet import self_net
# from model.sdn import self_net
from model.u2net import self_net
# from semseg.models.segformer import self_net
# from model.befunet import self_net
# from models.regseg import self_net
# from models.stdc import self_net
# from model.unet import *
# from models.segnext import *
# from UNet import *
# from models.pp_liteseg import *
from tqdm import tqdm
from util import *

# ------- 1. define loss function --------
import numpy as np
import random
import torch
MODEL_NAME = 'u2netp_AUG_b_l_D_b_s' 

if not os.path.exists('saved_models/' + MODEL_NAME):
    os.mkdir('saved_models/' + MODEL_NAME)
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 使用示例
set_seed(42)  # 设置随机种子为42
iouloss=IouLoss()
diceloss=DiceLoss()
Boundary_loss=BoundaryLoss()
bce_loss = nn.CrossEntropyLoss(size_average=True)

def muti_loss(outputs, labels):
    loss = 0.4 * bce_loss(outputs, labels.squeeze()) + \
           0.3 * iouloss(outputs, labels.squeeze()) + \
           0.1 * diceloss(outputs, labels.squeeze()) + \
           0.2 * Boundary_loss(outputs, labels.squeeze())
    return loss

def muti_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = muti_loss(d0, labels_v)
    loss1 = muti_loss(d1, labels_v)
    loss2 = muti_loss(d2, labels_v)
    loss3 = muti_loss(d3, labels_v)
    loss4 = muti_loss(d4, labels_v)
    loss5 = muti_loss(d5, labels_v)
    loss6 = muti_loss(d6, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss

model = self_net().cuda()
# model.load_state_dict(torch.load(r'saved_models\u2netp_AUG_b_l_D_b\u2netp_AUG_b_l_D_b_best_iou_0.8096_epoch_284.pth'))  # 加载权重

best_iou = 0.0  # To keep track of the best IoU
SEED = 42
LEARNING_RATE = 0.001
# SAVE_FREQ = 4000  # Save the model every 4000 iterations
NUM_EPOCHS = 500  # Define the number of epochs
T_MAX = NUM_EPOCHS  # For CosineAnnealingLR, typically T_max = NUM_EPOCHS
SAVE_DIR = os.path.join('saved_models', MODEL_NAME)

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("--- Starting Training ---")
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    ite_num = 0

    # Training Phase
    for data in tqdm(salobj_dataloader, total=len(salobj_dataloader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
        ite_num += 1
        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.long).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, d1, d2, d3, d4, d5, d6 = model(inputs)

        # Compute loss
        loss = muti_loss_fusion(outputs, d1, d2, d3, d4, d5, d6, labels.squeeze())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Save the model at specified frequency
        # if ite_num % SAVE_FREQ == 0:
        #     checkpoint_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_itr_{ite_num}_loss_{running_loss/ite_num:.4f}.pth")
        #     torch.save(model.state_dict(), checkpoint_path)
        #     print(f"Saved model checkpoint at iteration {ite_num} with loss {running_loss/ite_num:.4f}")
        #     running_loss = 0.0  # Reset running loss

    # Step the scheduler after each epoch
    scheduler.step()

    # Validation Phase
    model.eval()
    val_loss = 0.0
    sen_score = 0.0
    spe_score = 0.0
    acc_score = 0.0
    dsc_score = 0.0
    iou_score = 0.0

    with torch.no_grad():
        for data in tqdm(salobj_valdataloader, total=len(salobj_valdataloader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
            images, seg_masks = data['image'], data['label']

            images = images.type(torch.FloatTensor).to(device)
            seg_masks = seg_masks.type(torch.long).to(device)

            outputs, d1, d2, d3, d4, d5, d6 = model(images)
            loss = muti_loss_fusion(outputs, d1, d2, d3, d4, d5, d6, seg_masks.squeeze())
            
            val_loss += loss.item()

            # Compute metrics
            SEN, SPE, ACC, DSC, IoU = compute_metrics2(outputs, seg_masks)
            sen_score += SEN
            spe_score += SPE
            acc_score += ACC
            dsc_score += DSC
            iou_score += IoU

    # Average metrics over the validation set
    val_loss /= len(salobj_valdataloader)
    sen_score /= len(salobj_valdataloader)
    spe_score /= len(salobj_valdataloader)
    acc_score /= len(salobj_valdataloader)
    dsc_score /= len(salobj_valdataloader)
    iou_score /= len(salobj_valdataloader)

    # Check if current model is the best and save it
    mean_iou = np.mean(dsc_score[1:])  # Assuming dsc_score[1:] corresponds to class-wise IoU
    if mean_iou > best_iou:
        best_iou = mean_iou
        best_model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best_iou_{best_iou:.4f}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with Mean IoU: {best_iou:.4f}")

    # Print epoch statistics
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Validation Loss: {val_loss:.4f}, "
          f"Class1 IoU: {dsc_score[1]:.4f}, "
          f"Class2 IoU: {dsc_score[2]:.4f}, "
          f"Class3 IoU: {dsc_score[3]:.4f}, "
          f"Mean IoU: {mean_iou:.4f}, "
          f"Best IoU: {best_iou:.4f}")

print("--- Training Completed ---")

# 保存最终的输出结果到 jl.txt 文件
with open('jl.txt', 'w') as f:
    f.write(f"Training Completed\n")
    f.write(f"Final Validation Loss: {val_loss:.4f}\n")
    f.write(f"Final Class1 IoU: {dsc_score[1]:.4f}\n")
    f.write(f"Final Class2 IoU: {dsc_score[2]:.4f}\n")
    f.write(f"Final Class3 IoU: {dsc_score[3]:.4f}\n")
    f.write(f"Final Mean IoU: {mean_iou:.4f}\n")
    f.write(f"Final Best IoU: {best_iou:.4f}\n")