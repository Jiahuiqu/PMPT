import scipy.io as sio
import numpy as np




def load_data(dataset):
    if dataset == 1:
        train_HSI = r'./data_patch/Trento/HSI_data_tr.mat'
        train_LiDAR = r'./data_patch/Trento/LiDAR_data_tr.mat'
        train_label = r'./data_patch/Trento/TrLabel.mat'
        test_HSI = r'./data_patch/Trento/HSI_data_te.mat'
        test_LiDAR = r'./data_patch/Trento/LiDAR_data_te.mat'
        test_label = r'./data_patch/Trento/TeLabel.mat'

        train_image_HSI = sio.loadmat(train_HSI)['HSI_data_tr']
        train_image_LIDAR = sio.loadmat(train_LiDAR)['LiDAR_data_tr']
        train_label = sio.loadmat(train_label)['TrLabel']
        test_image_HSI = sio.loadmat(test_HSI)['HSI_data_te']
        test_image_LIDAR = sio.loadmat(test_LiDAR)['LiDAR_data_te']
        test_label = sio.loadmat(test_label)['TeLabel']
    elif dataset == 0:
        train_HSI = r'./data_patch/2013houston/HSI_data_tr.mat'
        train_LiDAR = r'./data_patch/2013houston/LiDAR_data_tr.mat'
        train_label = r'./data_patch/2013houston/TrLabel.mat'
        test_HSI = r'./data_patch/2013houston/HSI_data_te.mat'
        test_LiDAR = r'./data_patch/2013houston/LiDAR_data_te.mat'
        test_label = r'./data_patch/2013houston/TeLabel.mat'

        train_image_HSI = sio.loadmat(train_HSI)['HSI_data_tr']
        train_image_LIDAR = sio.loadmat(train_LiDAR)['LiDAR_data_tr']
        train_label = sio.loadmat(train_label)['TrLabel']
        test_image_HSI = sio.loadmat(test_HSI)['HSI_data_te']
        test_image_LIDAR = sio.loadmat(test_LiDAR)['LiDAR_data_te']
        test_label = sio.loadmat(test_label)['TeLabel']
    elif dataset == 2:
        train_HSI = r'./data_patch/Muufl/HSI_data_tr.mat'
        train_LiDAR = r'./data_patch/Muufl/LiDAR_data_tr.mat'
        train_label = r'./data_patch/Muufl/TrLabel.mat'
        test_HSI = r'./data_patch/Muufl/HSI_data_te.mat'
        test_LiDAR = r'./data_patch/Muufl/LiDAR_data_te.mat'
        test_label = r'./data_patch/Muufl/TeLabel.mat'

        train_image_HSI = sio.loadmat(train_HSI)['HSI_data_tr']
        train_image_LIDAR = sio.loadmat(train_LiDAR)['LiDAR_data_tr']
        train_label = sio.loadmat(train_label)['TrLabel']
        test_image_HSI = sio.loadmat(test_HSI)['HSI_data_te']
        test_image_LIDAR = sio.loadmat(test_LiDAR)['LiDAR_data_te']
        test_label = sio.loadmat(test_label)['TeLabel']
    elif dataset == 3:
        image_file_HSI = r'./datasets/Augsburg/HSI_data.mat'
        image_file_LiDAR = r'./datasets/Augsburg/LiDAR_data.mat'
        label_file = r'./datasets/Augsburg/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        label = label_data['All_Label']
    elif dataset == 'Augsburg_SAR':
        image_file_HSI = r'./datasets/Augsburg_SAR/HSI_data.mat'
        image_file_LiDAR = r'./datasets/Augsburg_SAR/SAR_data.mat'
        label_file = r'./datasets/Augsburg/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['SAR_data']
        label = label_data['All_Label']
    elif dataset == 'Berlin':
        image_file_HSI = r'./datasets/Berlin/HSI_data.mat'
        image_file_LiDAR = r'./datasets/Berlin/SAR_data_2.mat'
        label_file = r'./datasets/Berlin/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['SAR_data']
        label = label_data['All_Label']
    else:
        raise Exception('dataset does not find')

    train_image_HSI = train_image_HSI.astype(np.float32)
    train_image_LIDAR = train_image_LIDAR.astype(np.float32)
    train_label = train_label.astype(np.int64) - 1
    test_image_HSI = test_image_HSI.astype(np.float32)
    test_image_LIDAR = test_image_LIDAR.astype(np.float32)
    test_label = test_label.astype(np.int64) - 1
    test_label = np.squeeze(test_label)
    train_label = np.squeeze(train_label)
    return train_image_HSI, train_image_LIDAR, train_label, test_image_HSI, test_image_LIDAR, test_label
