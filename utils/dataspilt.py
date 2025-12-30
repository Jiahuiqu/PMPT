import numpy as np
import torch
import torch.nn as nn
import random
import time
import scipy.io as sio
from dataset_pre import get_dataset, data_standard, print_data, data_partition, gen_model_data, data_HSI_LIDATR
from sklearn.decomposition import PCA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#超参数

print('\n')
Seed_List = [1]   # 随机种子点

###################### 超参预设 ######################
curr_D_ratio = 100   # 每类训练集占这类总样本的比例，或每类训练样本的个数
# 的超参
patchsize_HSI = 11
patchsize_LiDAR = 11
batchsize = 300
LR = 0.01
FM = 30     # 输出的维度
BestAcc = 0     # 最优精度
m = 4
K = 10
D_num = 60

# 源域和目标域数据信息
def pca_whitening(image, number_of_pc):
    shape = image.shape

    image = np.reshape(image, [shape[0] * shape[1], shape[2]])
    number_of_rows = shape[0]
    number_of_columns = shape[1]
    pca = PCA(n_components=number_of_pc)
    image = pca.fit_transform(image)
    pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc), dtype=np.float32)
    for i in range(number_of_pc):
        pc_images[:, :, i] = np.reshape(image[:, i], (number_of_rows, number_of_columns))

    return pc_images

def adjust_learning_rate(LR, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['LR'] = LR

# ###################### 加载数据集 ######################
samples_type = ['ratio', 'same_num', 'fixed'][1]     # 训练集按照 0-按比例取训练集 1-按每类个数取训练集
# 选择数据集
datasets = 1
# dataset=1, 2012Houston
# dataset=2, trento
# 加载数据
[data_HSI, data_LiDAR, gt, train_gt, val_ratio, class_count, learning_rate,
 max_epoch, dataset_name, trainloss_result, LiDAR_bands] = get_dataset(datasets)



data_HSI = pca_whitening(data_HSI, number_of_pc=30)
height, width, bands = data_HSI.shape
# 数据标准化
[data_HSI, data_LiDAR] = data_standard(data_HSI, data_LiDAR, LiDAR_bands)
# 给LiDAR降一个维度
data_LiDAR = data_LiDAR[:, :, 0]

# 打印每类样本个数
print('#####源域样本个数#####')
print_data(gt, class_count)

# ###################### 参数初始化 ######################
train_samples_per_class = curr_D_ratio  # 当定义为每类样本个数时,则该参数更改为训练样本数
train_ratio = curr_D_ratio  # 训练比例

# ###################### 划分训练测试验证集 ######################
for curr_seed in Seed_List:
    random.seed(curr_seed)  # 当seed()没有参数时，每次生成的随机数是不一样的，而当seed()有参数时，每次生成的随机数是一样的

    # 对源域样本进行划分，得到训练、测试、验证集, 初始化D
    [train_label,test_label,D_label] = data_partition(samples_type, class_count, gt,train_gt, train_ratio, val_ratio, height, width)
    # ###################### 搭建网络 ######################
    # 搭建两个网络分别对HSI和LiDAR进行特征提取
    # sio.savemat("TrainImage.mat", {'TrainImage': train_label})

    [TrainPatch_HSI, TrainPatch_LiDAR,TrainLabel_HSI,TestPatch_HSI, TestPatch_LIDAR, TestLabel_HSI, D_Patch_HSI, D_Label_HSI,D_Patch_LiDAR, D_Label_LiDAR] = \
        gen_model_data(data_HSI, data_LiDAR, patchsize_HSI, patchsize_LiDAR,
                                           train_label, test_label, D_label, batchsize)


    sio.savemat("HSI_data_tr.mat", {'HSI_data_tr': TrainPatch_HSI.cpu().numpy()})
    sio.savemat("LiDAR_data_tr.mat", {'LiDAR_data_tr': TrainPatch_LiDAR.cpu().numpy()})
    sio.savemat("TrLabel.mat", {'TrLabel': TrainLabel_HSI.cpu().numpy()})
    sio.savemat("HSI_data_te.mat", {'HSI_data_te': TestPatch_HSI.cpu().numpy()})
    sio.savemat("LiDAR_data_te.mat", {'LiDAR_data_te': TestPatch_LIDAR.cpu().numpy()})
    sio.savemat("TeLabel.mat", {'TeLabel': TestLabel_HSI.cpu().numpy()})
    [train_data_HSI, D_data_HSI, train_data_LIDAR, D_data_LIDAR, test_data_HSI, test_data_LIDAR] = data_HSI_LIDATR(train_label, data_HSI, D_label, data_LiDAR, test_label)
    # sio.savemat("HSI_data_tr.mat", {'Data': TrainPatch_HSI.cpu().numpy()})
    # sio.savemat("LiDAR_data_tr.mat", {'Data': TrainPatch_LiDAR.cpu().numpy()})
    # sio.savemat("TrLabel.mat", {'Data': TrainLabel_HSI.cpu().numpy()})
    # sio.savemat("HSI_data_te.mat", {'Data': TestPatch_HSI.cpu().numpy()})
    # sio.savemat("LiDAR_data_te.mat", {'Data': TestPatch_LIDAR.cpu().numpy()})
    # sio.savemat("TeLabel.mat", {'Data': TestLabel_HSI.cpu().numpy()})