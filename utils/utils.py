import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import scipy as sp
import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
from skimage.transform import rotate
import torch.nn.functional as F
from augment import CenterResizeCrop


class Task(object):

    def __init__(self, data1, data2, num_classes, support_ratio, query_ratio):
        self.data1 = data1
        self.data2 = data2
        self.num_classes = num_classes
        self.support_ratio = support_ratio
        self.query_ratio = query_ratio

        class_folders = sorted(list(data1))

        # class_list = random.sample(class_folders, self.num_classes)
        class_list = class_folders
        labels = np.sort(class_list)
        # labels = np.array(range(len(class_list)))
        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas1 = []
        self.query_datas1 = []
        self.support_datas2 = []
        self.query_datas2 = []
        self.support_labels = []
        self.query_labels = []

        for c in class_list:
            temp1 = self.data1[c]  # list from data1
            temp2 = self.data2[c]  # list from data2
            samples[c] = list(zip(temp1, temp2))  # Combine samples pairwise

            random.shuffle(samples[c])

            total_samples_0 = len(self.data1[c])
            support_num = int(total_samples_0 * self.support_ratio)
            query_num = int(total_samples_0 * self.query_ratio)
            support_num = self.support_ratio
            query_num = self.query_ratio

            self.support_datas1 += [sample_pair[0] for sample_pair in samples[c][:support_num]]
            self.query_datas1 += [sample_pair[0] for sample_pair in samples[c][support_num:support_num + query_num]]
            # self.query_datas1 = self.support_datas1


            self.support_datas2 += [sample_pair[1] for sample_pair in samples[c][:support_num]]
            self.query_datas2 += [sample_pair[1] for sample_pair in samples[c][support_num:support_num + query_num]]
            # self.query_datas2 = self.support_datas2
            self.support_labels += [labels[c] for _ in range(support_num)]
            self.query_labels += [labels[c] for _ in range(query_num)]
            # self.query_labels = self.support_labels
        self.support_num = support_num
        self.query_num = query_num


class FewShotDataset(Dataset):
    def __init__(self, task, transformer = None, split='train'):
        self.task = task
        self.split = split
        self.transformer = transformer
        self.data = self.task.support_datas1 if self.split == 'train' else self.task.query_datas1
        self.data_LIDAR = self.task.support_datas2 if self.split == 'train' else self.task.query_datas2
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        label = self.labels[index]
        if self.transformer == None:
            img = torch.Tensor(np.asarray(self.data[index]))
            img_LIDAR = torch.Tensor(np.asarray(self.data_LIDAR[index]))
            return img, img_LIDAR, label
        elif len(self.transformer) == 2:
            img = torch.Tensor(np.asarray(self.transformer[1](self.transformer[0](self.data[index]))))
            img_LIDAR = torch.Tensor(np.asarray(self.transformer[1](self.transformer[0](self.data_LIDAR[index]))))
            return img, img_LIDAR, label
        else:
            img = torch.Tensor(np.asarray(self.transformer[0](self.data[index])))
            img_LIDAR = torch.Tensor(np.asarray(self.transformer[0](self.data_LIDAR[index])))
        # img = self.image_datas[index]
        # img_LIDAR = self.image_datas2[index]
        # label = self.labels[index]
        label = torch.tensor(label, dtype=torch.int64)
        return img, img_LIDAR, label


# Sampler
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:

            batch = [[i+j*self.num_per_class for i in torch.randperm(self.num_per_class)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_per_class for i in range(self.num_per_class)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

# dataloader
def get_HBKC_data_loader(task, num_per_class=1, scale=9, windowsize=11, split='train',shuffle = False):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和querya
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    transform_train = [CenterResizeCrop(scale_begin=scale, windowsize=windowsize)]
    dataset = HBKC_dataset(task, transformer=transform_train, split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, num_per_class*task.num_classes, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, num_per_class*task.num_classes, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=1024, sampler=sampler)

    return loader


def cosine_similarity(a, b):
    # Normalize the input tensors to unit vectors
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)

    # Compute the cosine similarity
    cosine_sim = torch.mm(a_norm, b_norm.t())

    return cosine_sim


def euclidean_metric(a, b):
    # n = a.shape[0]
    # m = b.shape[0]
    # a = a.unsqueeze(1).expand(n, m, -1)
    # b = b.unsqueeze(0).expand(n, m, -1)
    # logits = -((a - b)**2).sum(dim=2)
    # return logits

    cosine_sim = cosine_similarity(a, b)
    cosine_dist = 1 - cosine_sim
    return -cosine_dist

    # n = a.shape[0]
    # m = b.shape[0]
    # a = a.unsqueeze(1).expand(n, m, -1)
    # b = b.unsqueeze(0).expand(n, m, -1)
    # logits = -torch.abs(a - b).sum(dim=2)  # 没有负号，计算绝对差的和
    # return logits


def sup_constrive(representations, label, T):
    n = label.shape[0]
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    # similarity_matri = label.expand(n, n).eq(label.expand(n, n).t()).float()
    # 这步得到它的label矩阵，相同label的位置为1
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

    # 这步得到它的不同类的矩阵，不同类的位置为1
    mask_no_sim = torch.ones_like(mask) - mask
    # 这步产生一个对角线全为0的，其他位置为1的矩阵
    mask_dui_jiao_0 = torch.ones(n, n) - torch.eye(n, n)
    # 这步给相似度矩阵求exp,并且除以温度参数T
    similarity_matrix = torch.exp(similarity_matrix / T)
    # 这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
    similarity_matrix = similarity_matrix * mask_dui_jiao_0
    # 这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
    sim = mask * similarity_matrix
    # 用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
    no_sim = similarity_matrix - sim
    # 把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
    no_sim_sum = torch.sum(no_sim, dim=1)
    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)
    loss = mask_no_sim + loss + torch.eye(n, n)
    # 接下来就是算一个批次中的loss了
    loss = -torch.log(loss)  # 求-log
    # loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n
    # print(loss)  #0.9821
    # 最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
    loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

    return loss


def dict_storage(data_dict, new_tensor, new_labels):
    for i in range(len(new_labels)):
        a_label = new_labels[i].item()
        if a_label in data_dict.keys():
            # 如果标签已经存在，将新张量加到已存在的张量上，然后取平均
            # data_dict[a_label] = (data_dict[a_label] + new_tensor[i]) / 2
            data_dict[a_label] = new_tensor[i]
        else:
            # 如果标签不存在，将新标签和对应的张量添加到字典中
            data_dict[a_label] = new_tensor[i]
    return data_dict

def prompt_storage(data_dict, new_tensor, new_labels):
    a_label = new_labels
    data_dict[a_label] = new_tensor
    return data_dict


def meta_dataset(train_label, train_image_HSI):
    # keys_all_train = sorted(list(set(train_label)))
    # label_encoder_train = {}
    # for i in range(len(keys_all_train)):
    #     label_encoder_train[keys_all_train[i]] = i
    unique_labels = list(set(train_label))
    label_encoder_train = {label: label for index, label in enumerate(unique_labels)}
    # print(label_encoder_train)
    train_set_HSI = {}
    for class_, path in zip(train_label, train_image_HSI):
        if label_encoder_train[class_] not in train_set_HSI:
            train_set_HSI[label_encoder_train[class_]] = []
        train_set_HSI[label_encoder_train[class_]].append(path)
    return train_set_HSI


def rotate_image(image, angle):
    """旋转图像"""
    return rotate(image, angle, mode='wrap')


def augment_data(train_image_HSI, train_image_LIDAR, train_label):
    augmented_images = []
    augmented_images_LIDAR = []
    augmented_labels = []

    for i in range(len(train_image_HSI)):
        image = train_image_HSI[i]
        image_LIDAR = train_image_LIDAR[i]
        label = train_label[i]

        # 将原始图像和标签加入到数据集中
        augmented_images.append(image)
        augmented_images_LIDAR.append(image_LIDAR)
        augmented_labels.append(label)

        # 对图像进行90, 180, 270度旋转，生成新的图像和标签
        for angle in [90, 180, 270]:
            rotated_image = rotate_image(image, angle)
            rotated_image_LIDAR = rotate_image(image_LIDAR, angle)
            augmented_images.append(rotated_image)
            augmented_images_LIDAR.append(rotated_image_LIDAR)
            augmented_labels.append(label)

    # 转换为 numpy 数组
    augmented_images = np.array(augmented_images)
    augmented_images_LIDAR = np.array(augmented_images_LIDAR)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_images_LIDAR, augmented_labels