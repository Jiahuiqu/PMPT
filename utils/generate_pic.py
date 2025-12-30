import argparse
import random
import scipy.io as sio
import numpy as np
import torch
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.io import savemat
from data_read import get_val_num_test, read_test_data

indianpines_colors = np.array([
    [0, 0, 0],          # Unclassified (空)
    [34, 139, 34],      # Healthy Grass (森林绿)
    [107, 142, 35],     # Stressed Grass (橄榄绿)
    [0, 100, 0],        # Artificial turf (深绿)
    [0, 128, 0],        # Evergreen trees (绿色)
    [50, 205, 50],      # Deciduous trees (森林绿)
    [255, 128, 0],      # Bare earth (棕褐色)
    [0, 191, 255],      # Water (深天蓝)
    [255, 222, 173],    # Residential buildings (浅肉色)
    [218, 165, 32],     # Non-residential buildings (金黄)
    [128, 128, 128],    # Roads (灰色)
    [192, 192, 192],    # Sidewalks (银色)
    [255, 69, 0],       # Crosswalks (橙红色)
    [178, 34, 34],      # Major thoroughfares (深红)
    [139, 0, 0],        # Highways (深红)
    [70, 130, 180],     # Railways (钢蓝)
    [128, 0, 128],      # Paved parking lots (紫色)
    [139, 69, 19],      # Unpaved parking lots (马鞍棕)
    [255, 0, 0],        # Cars (红色)
    [0, 0, 255],        # Trains (蓝色)
    [255, 255, 0]       # Stadium seats (黄色)
])
indianpines_colors = preprocessing.minmax_scale(indianpines_colors, feature_range=(0, 1))

def classification_map(img, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    # ax.imshow(img)
    fig.savefig(save_path, dpi=dpi)
    
    return 0 


def generate(image, gt, index, nTrain_perClass, nvalid_perClass, test_pred, halfsize):
    number_of_rows = np.size(image,0)
    number_of_columns = np.size(image,1)

    predicted_thematic_map = np.zeros(shape=(number_of_rows, number_of_columns))
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            predicted_thematic_map[i, j] = gt[i,j]
    nclass = np.max(gt)
    
    fl = 0
    for i in range(nclass):
        print('test lable of class:',i)
        matrix = index[i]
        temprow = matrix[:,0]
        tempcol = matrix[:,1]
        m = len(temprow)
        fl = fl - nTrain_perClass[i] - nvalid_perClass[i]
        for j in range(nTrain_perClass[i] + nvalid_perClass[i], m):
            predicted_thematic_map[temprow[j], tempcol[j]] = test_pred.squeeze()[fl + j]+1
        fl = fl + m

    predicted_thematic_map = predicted_thematic_map[halfsize:number_of_rows -halfsize,halfsize:number_of_columns-halfsize]

    path = '.'
    # classification_map(predicted_thematic_map, gt, 600,
    #                     path + '/classification_maps/' + dataset + '_' + day_str +'_' + str(num) +'_OA_'+ str(round(OA, 2)) + '_' + model_name +  '.png')
    
    return predicted_thematic_map

def main(args):
    print(args)
    args.num_of_ex = 1
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(1)  # 设置使用 GPU 1
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    for session in range(args.sessions):
        if session == 0:
            image_file = r'./result/2018houston/session0_0.871549865732395.mat'
            pred_array = sio.loadmat(image_file)['map']
        elif session == 1:
            image_file = r'./result/2018houston/session1_0.7602613878028152.mat'
            pred_array = sio.loadmat(image_file)['map']
        elif session == 2:
            image_file = r'./result/2018houston/session2_0.7104450212464305.mat'
            pred_array = sio.loadmat(image_file)['map']
        elif session == 3:
            image_file = r'./result/2018houston/session3_0.6919993095960089.mat'
            pred_array = sio.loadmat(image_file)['map']
        test_base(session, 0, pred_array, args)


def test_base(session, num, pred_array, args):
    num_of_samples = get_val_num_test(session, args.ratio, args.dataset)
    train_image_HSI, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR, validation_label, nTrain_perClass, nvalid_perClass, \
    train_index, val_index, index, image, image_LIDAR, gt, s = read_test_data(args.type, session, args.windowsize,
                                                                        args.old_num_perclass, num_of_samples, num,
                                                                        args.dataset)
    print("Testing ...")
    halfsize = int(args.windowsize / 2)


    classification_map = generate(image, gt, index, nTrain_perClass, nvalid_perClass, pred_array, halfsize)
    savemat('result/' + args.dataset  + '/session'+ str(session) + '_' + '.mat', {'map': classification_map})

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Pre training
    parser.add_argument('--old_num_perclass', type=int, default=180)
    parser.add_argument('--new_num_perclass', type=int, default=5)

    # Base
    parser.add_argument('--SHOT_RATIO_PER_CLASS', type=int, default=60)
    parser.add_argument('--QUERY_RATIO_PER_CLASS', type=int, default=120)

    # Incremental
    parser.add_argument('--Incremental_SHOT_RATIO_PER_CLASS', type=int, default=12)
    parser.add_argument('--Incremental_QUERY_RATIO_PER_CLASS', type=int, default=8)

    parser.add_argument('--ratio', default=0.1, type=float,
                        help='ratio of val (default: 0.1)')
    # Pre training
    parser.add_argument('--seed', dest='seed', default=114514, type=int,
                        help='Random seed')
    parser.add_argument('--windowsize', type=int, default=11)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=301)
    parser.add_argument('--fine_epochs', type=int, default=101)
    parser.add_argument('--sessions', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='2018houston')
    parser.add_argument('--pre_lr', type=float, default=1e-3)
    parser.add_argument('--task_lr', type=float, default=0.1)
    parser.add_argument('--type', type=str, default='none')
    # Augmentation
    parser.add_argument('--augment', default=True, type=bool,
                        help='either use data augmentation or not (default: False)')
    parser.add_argument('--scale', default=9, type=int,
                        help='the minimum scale for center crop (default: 19)')

    # encoder specifics
    parser.add_argument('--encoder_dim', default=128, type=int,
                        help='feature dimension for encoder (default: 64)')
    parser.add_argument('--encoder_depth', default=4, type=int,
                        help='encoder_depth; number of blocks ')
    parser.add_argument('--encoder_num_heads', default=8, type=int,
                        help='number of heads of encoder (default: 8)')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    main(args)

