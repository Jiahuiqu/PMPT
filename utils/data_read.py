import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import h5py


def pca_whitening(image, number_of_pc):

    shape = image.shape
    
    image = np.reshape(image, [shape[0]*shape[1], shape[2]])
    number_of_rows = shape[0]
    number_of_columns = shape[1]
    pca = PCA(n_components = number_of_pc)
    image = pca.fit_transform(image)
    pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc),dtype=np.float32)
    for i in range(number_of_pc):
        pc_images[:, :, i] = np.reshape(image[:, i], (number_of_rows, number_of_columns))
    
    return pc_images

def load_data(dataset, session):
    if dataset == 'Trento':
        image_file_HSI = r'./datasets/Trento/HSI_data.mat'
        image_file_LiDAR = r'./datasets/Trento/LiDAR_data.mat'
        if session == 0:
            label_file = r'./datasets/Trento/All_Label(1-2).mat'
        elif session == 1:
            label_file = r'./datasets/Trento/All_Label(3-4).mat'
        elif session == 2:
            label_file = r'./datasets/Trento/All_Label(5-6).mat'
        else:
            label_file = r'./datasets/Trento/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        label = label_data['All_Label']
    elif dataset == '2013houston':
        image_file_HSI = r'./datasets/2013houston/HSI_data.mat'
        image_file_LiDAR = r'./datasets/2013houston/LiDAR_data.mat'
        if session == 0:
            label_file = r'./datasets/2013houston/All_Label(1-9).mat'
        elif session == 1:
            label_file = r'./datasets/2013houston/All_Label(10-12).mat'
        elif session == 2:
            label_file = r'./datasets/2013houston/All_Label(13-15).mat'
        else:
            label_file = r'./datasets/2013houston/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)        
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        label = label_data['All_Label']
    elif dataset == '2018houston':
        image_file_HSI = r'./datasets/2018houston/HSI_data.mat'
        image_file_LiDAR = r'./datasets/2018houston/LiDAR_data.mat'
        if session == 0:
            label_file = r'./datasets/2018houston/All_Label(1-11).mat'
        elif session == 1:
            label_file = r'./datasets/2018houston/All_Label(12-14).mat'
        elif session == 2:
            label_file = r'./datasets/2018houston/All_Label(15-17).mat'
        elif session == 3:
            label_file = r'./datasets/2018houston/All_Label(18-20).mat'
        else:
            label_file = r'./datasets/2018houston/All_Label.mat'

        image_data_HSI =  h5py.File(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data =  sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data'][:]
        image_HSI = np.transpose(image_HSI, (2, 1, 0))
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        label = label_data['All_Label']

    elif dataset == 'Muufl':
        image_file_HSI = r'./datasets/Muufl/HSI_data.mat'
        image_file_LiDAR = r'./datasets/Muufl/LiDAR_data_1.mat'
        label_file = r'./datasets/Muufl/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        label = label_data['All_Label']
    elif dataset == 'Augsburg':
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
        image_file_LiDAR = r'./datasets/Berlin/SAR_data.mat'
        if session == 0:
            label_file = r'./datasets/Berlin/All_Label(1-4).mat'
        elif session == 1:
            label_file = r'./datasets/Berlin/All_Label(5-6).mat'
        elif session == 2:
            label_file = r'./datasets/Berlin/All_Label(7-8).mat'
        else:
            label_file = r'./datasets/Berlin/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['SAR_data']
        label = label_data['All_Label']
    else:
        raise Exception('dataset does not find')
    image_HSI = image_HSI.astype(np.float32)
    image_LiDAR = image_LiDAR.astype(np.float32)
    label = label.astype(np.int64)
    return image_HSI, image_LiDAR, label


def readdata(type, session, windowsize, train_num, val_num, num, dataset):
    or_image_HSI, or_image_LiDAR, or_label = load_data(dataset, session)
    # image = np.expand_dims(image, 2)
    halfsize = int((windowsize - 1) / 2)
    number_class = (np.max(or_label) - np.min(or_label[or_label != 0]) + 1).astype(np.int64)
    if dataset == 'Augsburg_SAR':
        pass
    elif dataset == 'Berlin':
        pass
    else:
        or_image_LiDAR = np.expand_dims(or_image_LiDAR, 2)
    # or_image_LiDAR = np.expand_dims(or_image_LiDAR, 2)
    image = np.pad(or_image_HSI, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
    image_LiDAR = np.pad(or_image_LiDAR, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
    label = np.pad(or_label, ((halfsize, halfsize), (halfsize, halfsize)), 'constant', constant_values=0)

    if type == 'PCA':
        image1 = pca_whitening(image, number_of_pc=30)
        image_LiDAR1 = np.copy(image_LiDAR)
    elif type == 'none':
        image1 = np.copy(image)
        image_LiDAR1 = np.copy(image_LiDAR)
    else:
        raise Exception('type does not find')

    image = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image_LiDAR = (image_LiDAR1 - np.min(image_LiDAR1)) / (np.max(image_LiDAR1) - np.min(image_LiDAR1))
    # set the manner of selecting training samples

    n = np.zeros(number_class, dtype=np.int64)
    label_class = list(set(label[label != 0]))
    label_class.sort()
    for i, j in zip(label_class, range(number_class)):
        temprow, tempcol = np.where(label == i)
        n[j] = len(temprow)
    total_num = np.sum(n)

    nTrain_perClass = np.ones(number_class, dtype=np.int64) * train_num
    for i in range(number_class):
        if n[i] <= nTrain_perClass[i]:
            nTrain_perClass[i] = 5
    ###验证机数目
    nValidation_perClass = (n / total_num) * val_num
    nvalid_perClass = nValidation_perClass.astype(np.int32)

    index = []
    flag = 0
    fl = 0

    bands = np.size(image, 2)
    bands_LIDAR = np.size(image_LiDAR, 2)
    validation_image = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, bands], dtype=np.float32)
    validation_image_LIDAR = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, bands_LIDAR], dtype=np.float32)
    validation_label = np.zeros(np.sum(nvalid_perClass), dtype=np.int64)
    train_image = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, bands], dtype=np.float32)
    train_image_LIDAR = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, bands_LIDAR], dtype=np.float32)
    train_label = np.zeros(np.sum(nTrain_perClass), dtype=np.int64)
    train_index = np.zeros([np.sum(nTrain_perClass), 2], dtype=np.int32)
    val_index = np.zeros([np.sum(nvalid_perClass), 2], dtype=np.int32)

    for k, i in zip(label_class, range(number_class)):
        temprow, tempcol = np.where(label == k)

        matrix = np.zeros([len(temprow), 2], dtype=np.int64)
        matrix[:, 0] = temprow
        matrix[:, 1] = tempcol
        np.random.seed(num)
        np.random.shuffle(matrix)

        temprow = matrix[:, 0]
        tempcol = matrix[:, 1]
        index.append(matrix)

        for j in range(nTrain_perClass[i]):
            train_image[flag + j, :, :, :] = image[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                             (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_image_LIDAR[flag + j, :, :, :] = image_LiDAR[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                                   (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_label[flag + j] = k - 1
            train_index[flag + j] = matrix[j, :]
        flag = flag + nTrain_perClass[i]

        for j in range(nTrain_perClass[i], nTrain_perClass[i] + nvalid_perClass[i]):
            validation_image[fl + j - nTrain_perClass[i], :, :, :] = image[(temprow[j] - halfsize):(
                        temprow[j] + halfsize + 1),
                                                                     (tempcol[j] - halfsize):(
                                                                                 tempcol[j] + halfsize + 1)]
            validation_image_LIDAR[fl + j - nTrain_perClass[i], :, :, :] = image_LiDAR[(temprow[j] - halfsize):(
                        temprow[j] + halfsize + 1),
                                                                           (tempcol[j] - halfsize):(
                                                                                       tempcol[j] + halfsize + 1)]
            validation_label[fl + j - nTrain_perClass[i]] = k - 1
            val_index[fl + j - nTrain_perClass[i]] = matrix[j, :]
        fl = fl + nvalid_perClass[i]


    train_image = np.transpose(train_image,(0,3,1,2))
    train_image_LIDAR = np.transpose(train_image_LIDAR,(0,3,1,2))

    validation_image = np.transpose(validation_image,(0,3,1,2))
    validation_image_LIDAR = np.transpose(validation_image_LIDAR,(0,3,1,2))
    return train_image, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR, validation_label, \
           nTrain_perClass, nvalid_perClass, train_index, val_index, index, image, image_LiDAR, label, total_num


def get_val_num(session, ratio, dataset):

    or_image_HSI, or_image_LiDAR, or_label = load_data(dataset, session)

    number_class = (np.max(or_label) - np.min(or_label[or_label != 0]) + 1).astype(np.int64)
    n = np.zeros(number_class,dtype=np.int64)
    label_class = list(set(or_label[or_label != 0]))
    label_class.sort()
    for i, j in zip(label_class, range(number_class)):
        temprow, tempcol = np.where(or_label == i)
        n[j] = len(temprow)
    total_num = np.sum(n)

    return total_num * ratio

def get_val_num_test(session, ratio, dataset):

    or_image_HSI, or_image_LiDAR, or_label = load_test_data(dataset, session)

    number_class = np.max(or_label).astype(np.int64)
    n = np.zeros(number_class,dtype=np.int64)
    for i in range(number_class):
        temprow, tempcol = np.where(or_label == i + 1)
        n[i] = len(temprow)
    total_num = np.sum(n)

    return total_num * ratio


def read_test_data(type, session, windowsize, train_num, val_num, num, dataset):
    or_image_HSI, or_image_LiDAR, or_label = load_test_data(dataset, session)
    # image = np.expand_dims(image, 2)
    halfsize = int((windowsize - 1) / 2)
    number_class = (np.max(or_label) - np.min(or_label[or_label != 0]) + 1).astype(np.int64)
    if dataset == 'Augsburg_SAR':
        pass
    elif dataset == 'Berlin':
        pass
    else:
        or_image_LiDAR = np.expand_dims(or_image_LiDAR, 2)
    image = np.pad(or_image_HSI, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
    image_LiDAR = np.pad(or_image_LiDAR, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
    label = np.pad(or_label, ((halfsize, halfsize), (halfsize, halfsize)), 'constant', constant_values=0)

    if type == 'PCA':
        image1 = pca_whitening(image, number_of_pc=30)
        image_LiDAR1 = np.copy(image_LiDAR)
    elif type == 'none':
        image1 = np.copy(image)
        image_LiDAR1 = np.copy(image_LiDAR)
    else:
        raise Exception('type does not find')

    image = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image_LiDAR = (image_LiDAR1 - np.min(image_LiDAR1)) / (np.max(image_LiDAR1) - np.min(image_LiDAR1))
    # set the manner of selecting training samples

    n = np.zeros(number_class, dtype=np.int64)
    for i in range(number_class):
        temprow, tempcol = np.where(label == i + 1)
        n[i] = len(temprow)
    total_num = np.sum(n)

    nTrain_perClass = np.ones(number_class, dtype=np.int64) * train_num
    for i in range(number_class):
        if n[i] <= nTrain_perClass[i]:
            nTrain_perClass[i] = 5
    ###验证机数目
    nValidation_perClass = (n / total_num) * val_num
    nvalid_perClass = nValidation_perClass.astype(np.int32)

    index = []
    flag = 0
    fl = 0

    bands = np.size(image, 2)
    bands_LIDAR = np.size(image_LiDAR, 2)
    validation_image = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, bands], dtype=np.float32)
    validation_image_LIDAR = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, bands_LIDAR], dtype=np.float32)
    validation_label = np.zeros(np.sum(nvalid_perClass), dtype=np.int64)
    train_image = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, bands], dtype=np.float32)
    train_image_LIDAR = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, bands_LIDAR], dtype=np.float32)
    train_label = np.zeros(np.sum(nTrain_perClass), dtype=np.int64)
    train_index = np.zeros([np.sum(nTrain_perClass), 2], dtype=np.int32)
    val_index = np.zeros([np.sum(nvalid_perClass), 2], dtype=np.int32)

    for i in range(number_class):
        temprow, tempcol = np.where(label == i + 1)

        matrix = np.zeros([len(temprow), 2], dtype=np.int64)
        matrix[:, 0] = temprow
        matrix[:, 1] = tempcol
        np.random.seed(num)
        np.random.shuffle(matrix)

        temprow = matrix[:, 0]
        tempcol = matrix[:, 1]
        index.append(matrix)

        for j in range(nTrain_perClass[i]):
            train_image[flag + j, :, :, :] = image[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                             (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_image_LIDAR[flag + j, :, :, :] = image_LiDAR[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                                   (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_label[flag + j] = i
            train_index[flag + j] = matrix[j, :]
        flag = flag + nTrain_perClass[i]

        for j in range(nTrain_perClass[i], nTrain_perClass[i] + nvalid_perClass[i]):
            validation_image[fl + j - nTrain_perClass[i], :, :, :] = image[(temprow[j] - halfsize):(
                        temprow[j] + halfsize + 1),
                                                                     (tempcol[j] - halfsize):(
                                                                                 tempcol[j] + halfsize + 1)]
            validation_image_LIDAR[fl + j - nTrain_perClass[i], :, :, :] = image_LiDAR[(temprow[j] - halfsize):(
                        temprow[j] + halfsize + 1),
                                                                           (tempcol[j] - halfsize):(
                                                                                       tempcol[j] + halfsize + 1)]
            validation_label[fl + j - nTrain_perClass[i]] = i
            val_index[fl + j - nTrain_perClass[i]] = matrix[j, :]
        fl = fl + nvalid_perClass[i]


    train_image = np.transpose(train_image,(0,3,1,2))
    train_image_LIDAR = np.transpose(train_image_LIDAR,(0,3,1,2))

    validation_image = np.transpose(validation_image,(0,3,1,2))
    validation_image_LIDAR = np.transpose(validation_image_LIDAR,(0,3,1,2))
    return train_image, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR, validation_label, \
           nTrain_perClass, nvalid_perClass, train_index, val_index, index, image, image_LiDAR, label, total_num


def load_test_data(dataset, session):
    if dataset == 'Trento':
        image_file_HSI = r'./datasets/Trento/HSI_data.mat'
        image_file_LiDAR = r'./datasets/Trento/LiDAR_data.mat'
        if session == 0:
            label_file = r'./datasets/Trento/All_Label(1-2).mat'
        elif session == 1:
            label_file = r'./datasets/Trento/All_Label(1-4).mat'
        else:
            label_file = r'./datasets/Trento/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        label = label_data['All_Label']
    elif dataset == '2013houston':
        image_file_HSI = r'./datasets/2013houston/HSI_data.mat'
        image_file_LiDAR = r'./datasets/2013houston/LiDAR_data.mat'
        if session == 0:
            label_file = r'./datasets/2013houston/All_Label(1-9).mat'
        elif session == 1:
            label_file = r'./datasets/2013houston/All_Label(1-12).mat'
        else:
            label_file = r'./datasets/2013houston/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        label = label_data['All_Label']
    elif dataset == '2018houston':
        image_file_HSI = r'./datasets/2018houston/HSI_data.mat'
        image_file_LiDAR = r'./datasets/2018houston/LiDAR_data.mat'
        if session == 0:
            label_file = r'./datasets/2018houston/All_Label(1-11).mat'
        elif session == 1:
            label_file = r'./datasets/2018houston/All_Label(1-14).mat'
        elif session == 2:
            label_file = r'./datasets/2018houston/All_Label(1-17).mat'
        else:
            label_file = r'./datasets/2018houston/All_Label.mat'
        image_data_HSI =  h5py.File(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data =  sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data'][:]
        image_HSI = np.transpose(image_HSI, (2, 1, 0))
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        label = label_data['All_Label']
    elif dataset == 'Muufl':
        image_file_HSI = r'./datasets/Muufl/HSI_data.mat'
        image_file_LiDAR = r'./datasets/Muufl/LiDAR_data_1.mat'
        label_file = r'./datasets/Muufl/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        label = label_data['All_Label']
    elif dataset == 'Augsburg':
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
        image_file_LiDAR = r'./datasets/Berlin/SAR_data.mat'
        if session == 0:
            label_file = r'./datasets/Berlin/All_Label(1-4).mat'
        elif session == 1:
            label_file = r'./datasets/Berlin/All_Label(1-6).mat'
        else:
            label_file = r'./datasets/Berlin/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['SAR_data']
        label = label_data['All_Label']
    else:
        raise Exception('dataset does not find')
    image_HSI = image_HSI.astype(np.float32)
    image_LiDAR = image_LiDAR.astype(np.float32)
    label = label.astype(np.int64)
    return image_HSI, image_LiDAR, label