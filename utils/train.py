import argparse
import torch
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import TensorDataset, DataLoader
import utils
from Data_preprocessing.read_data import load_data
from augment import CenterResizeCrop
from data_read import readdata, get_val_num
from hyper_dataset import HyperData
from model import vit_HSI_LIDAR_patch3
from sklearn import metrics
from collections import OrderedDict
import torch.nn.functional as F
from suploss import ProxyPLoss


def train(args):
    for num in range(0, args.num_of_ex):
        proto_dict = {}
        for session in range(args.session):
            if session == 0:
                #######################           Model initialization           #######################
                feature_encoder = vit_HSI_LIDAR_patch3(img_size=(args.windowsize, args.windowsize), in_chans=30,
                                                       in_chans_LIDAR=1,
                                                       hid_chans=128, hid_chans_LIDAR=128, embed_dim=args.encoder_dim,
                                                       depth=args.encoder_depth,
                                                       num_heads=args.encoder_num_heads, mlp_ratio=2.0, num_classes=6,
                                                       global_pool=False)
                feature_encoder.cuda()
                #######################           optimizer,scheduler,loss           #######################
                optimizer = torch.optim.Adam(feature_encoder.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
                crossEntropy = torch.nn.CrossEntropyLoss().cuda()
                # pcl = ProxyPLoss(num_classes=number_class, scale=12).cuda()

                #######################           Data loading and preparation           #######################
                train_image_HSI, train_image_LIDAR, train_label, test_image_HSI, test_image_LIDAR, test_label = load_data(session)

                random_numbers = random.sample(range(200), 100)
                train_label = np.delete(train_label, random_numbers)
                train_image_HSI = np.delete(train_image_HSI, random_numbers, axis=0)
                train_image_LIDAR = np.delete(train_image_LIDAR, random_numbers, axis=0)
                number_class = len(np.unique(train_label))

                train_set_HSI = utils.meta_dataset(train_label, train_image_HSI)
                train_set_LiDAR = utils.meta_dataset(train_label, train_image_LIDAR)

                transform_train = [CenterResizeCrop(scale_begin=args.scale, windowsize=args.windowsize)]
                train_dataset = HyperData((train_image_HSI, train_image_LIDAR, train_label), transform_train)
                test_dataset = TensorDataset(torch.tensor(test_image_HSI), torch.tensor(test_image_LIDAR),
                                              torch.tensor(test_label))
                train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
                test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False)

                for epoch in range(args.epochs):

                    feature_encoder.train()
                    total_loss = 0
                    for idx, (HSI_data, LiDAR_data, label) in enumerate(train_loader):
                        HSI_data, LiDAR_data, label = HSI_data.cuda(), LiDAR_data.cuda(), label.cuda()
                        task = utils.Task(train_set_HSI, train_set_LiDAR, number_class, args.SHOT_RATIO_PER_CLASS,
                                              args.QUERY_RATIO_PER_CLASS)
                        SHOT_NUM_PER_CLASS = int(len(task.support_labels) / number_class)
                        support_dataloader = utils.get_HBKC_data_loader(task,
                                                                        num_per_class=SHOT_NUM_PER_CLASS,
                                                                        scale = args.scale, windowsize = args.windowsize,
                                                                        split="train", shuffle=False)
                        QUERY_NUM_PER_CLASS = int(len(task.query_labels) / number_class)
                        query_dataloader = utils.get_HBKC_data_loader(task,
                                                                      num_per_class=QUERY_NUM_PER_CLASS,
                                                                      scale=args.scale, windowsize=args.windowsize,
                                                                      split="test", shuffle=True)
                        supports_HSI, supports_LiDAR, support_labels = next(support_dataloader.__iter__())  # (batchsize, chan, 11, 11)
                        supports_HSI, supports_LiDAR, support_labels = supports_HSI.cuda(), supports_LiDAR.cuda(), support_labels.cuda()
                        querys_HSI, querys_LiDAR, query_labels = next(query_dataloader.__iter__())  # (batchsize, chan, 11, 11)
                        querys_HSI, querys_LiDAR, query_labels = querys_HSI.cuda(), querys_LiDAR.cuda(), query_labels.cuda()

                        #######################           Prototype calculation process of support set           #######################
                        support_features = feature_encoder(supports_HSI, supports_LiDAR)
                        support_proto = support_features.reshape(number_class, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
                        proto_labels = torch.unique(support_labels)

                        #######################           Overall prototype estimate           #######################
                        tmp = support_proto.detach()
                        tmp_label = proto_labels.long()
                        proto_dict = utils.dict_storage(proto_dict, tmp, tmp_label, new_session=False)
                        all_proto = torch.stack(list(proto_dict.values()))
                        all_proto_label = torch.tensor(list(proto_dict.keys())).cuda()

                        #######################           Classification loss calculation process for support set           #######################
                        logits_support = utils.euclidean_metric(support_features, all_proto)
                        support_cls_loss = crossEntropy(logits_support, support_labels.long().cuda())

                        #######################           Classification loss calculation process for query set           #######################
                        query_features = feature_encoder(querys_HSI, querys_LiDAR)
                        logits_query = utils.euclidean_metric(query_features, all_proto)
                        query_cls_loss = crossEntropy(logits_query, query_labels.long().cuda())

                        #######################           Supervised comparison loss calculation process           #######################
                        features = feature_encoder(HSI_data, LiDAR_data)
                        cos_features = torch.concat((features, all_proto),dim=0).cpu()
                        combined_labels = torch.concat((label, all_proto_label), dim=0).cpu()
                        # loss_con = pcl(features, label, all_proto)
                        # loss_con = utils.sup_constrive(features.cpu(), label.cpu(), T = 0.09)
                        loss_con = utils.sup_constrive(cos_features.cpu(), combined_labels.cpu(), T = 0.09)

                        #######################           Model gradient backpropagation           #######################
                        loss_all = support_cls_loss + query_cls_loss + loss_con * 0.05
                        feature_encoder.zero_grad()
                        loss_all.backward()
                        optimizer.step()
                        total_loss = total_loss + loss_all

                    scheduler.step()
                    total_loss = total_loss / (idx + 1)
                    print('epoch:', epoch,
                          'loss:', total_loss.data.cpu().numpy())
                    state = {'model': feature_encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

                #######################           Model storage and testing           #######################
                torch.save(state, './model/base_model.pt')
                print("Testing session_0...")
                session0_true_cla, session0_overall_accuracy, session0_average_accuracy, session0_kappa, session0_true_label, \
                session0_test_pred, session0_cm =test_batch(feature_encoder, train_loader, test_loader, proto_dict)
                print('session_0:')
                print('overall_accuracy: {0:f}'.format(session0_overall_accuracy),
                      'average_accuracy: {0:f}'.format(session0_average_accuracy),
                      'kappa:{0:f}'.format(session0_kappa))

            else:
                model = vit_HSI_LIDAR_patch3(img_size=(args.windowsize, args.windowsize), in_chans=30,
                                                       in_chans_LIDAR=1,
                                                       hid_chans=128, hid_chans_LIDAR=128, embed_dim=args.encoder_dim,
                                                       depth=args.encoder_depth,
                                                       num_heads=args.encoder_num_heads, mlp_ratio=2.0, num_classes=6,
                                                       global_pool=False)
                if session == 1:
                    checkpoint_model = torch.load('./model/base_model.pt')['model']
                    model.load_state_dict(checkpoint_model, strict=False)
                    # 自定义冻结部分参数
                    for name, parameter in model.named_parameters():
                        if 'prompt' not in name:
                            parameter.requries_grad = False
                else:
                    checkpoint_model = torch.load('./model/session'+ str(session-1) +'_model.pt')['model']
                    model.load_state_dict(checkpoint_model, strict=False)

                # 过滤传入优化器的参数
                # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, identifier.parameters()))

                model.cuda()
                #######################           optimizer,scheduler,loss           #######################
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
                crossEntropy = torch.nn.CrossEntropyLoss().cuda()

                #######################           Data loading and preparation           #######################
                train_image_HSI, train_image_LIDAR, train_label, test_image_HSI, test_image_LIDAR, test_label = load_data(session)
                number_class = len(np.unique(train_label))
                train_set_HSI = utils.meta_dataset(train_label, train_image_HSI)
                train_set_LiDAR = utils.meta_dataset(train_label, train_image_LIDAR)
                transform_train = [CenterResizeCrop(scale_begin=args.scale, windowsize=args.windowsize)]
                train_dataset = HyperData((train_image_HSI, train_image_LIDAR, train_label), transform_train)
                train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
                if session == 1:
                    test_dataset_session_1 = TensorDataset(torch.tensor(test_image_HSI), torch.tensor(test_image_LIDAR),
                                              torch.tensor(test_label))
                    test_loader_session_1 = DataLoader(dataset=test_dataset_session_1, batch_size=args.batch_size, shuffle=False)
                if session == 2:
                    test_dataset_session_2 = TensorDataset(torch.tensor(test_image_HSI), torch.tensor(test_image_LIDAR),
                                              torch.tensor(test_label))
                    test_loader_session_2 = DataLoader(dataset=test_dataset_session_2, batch_size=args.batch_size, shuffle=False)


                for epoch in range(args.fine_epochs):
                    model.train()
                    total_loss = 0
                    for idx, (HSI_data, LiDAR_data, label) in enumerate(train_loader):
                        HSI_data, LiDAR_data, label = HSI_data.cuda(), LiDAR_data.cuda(), label.cuda()
                        task = utils.Task(train_set_HSI, train_set_LiDAR, number_class, args.SHOT_RATIO_PER_CLASS,
                                              args.QUERY_RATIO_PER_CLASS)
                        SHOT_NUM_PER_CLASS = int(len(task.support_labels) / number_class)
                        support_dataloader = utils.get_HBKC_data_loader(task,
                                                                        num_per_class=SHOT_NUM_PER_CLASS,
                                                                        scale = args.scale, windowsize = args.windowsize,
                                                                        split="train", shuffle=False)
                        QUERY_NUM_PER_CLASS = int(len(task.query_labels) / number_class)
                        query_dataloader = utils.get_HBKC_data_loader(task,
                                                                      num_per_class=QUERY_NUM_PER_CLASS,
                                                                      scale=args.scale, windowsize=args.windowsize,
                                                                      split="test", shuffle=True)
                        supports_HSI, supports_LiDAR, support_labels = next(support_dataloader.__iter__())  # (batchsize, chan, 11, 11)
                        supports_HSI, supports_LiDAR, support_labels = supports_HSI.cuda(), supports_LiDAR.cuda(), support_labels.cuda()
                        querys_HSI, querys_LiDAR, query_labels = next(query_dataloader.__iter__())  # (batchsize, chan, 11, 11)
                        querys_HSI, querys_LiDAR, query_labels = querys_HSI.cuda(), querys_LiDAR.cuda(), query_labels.cuda()

                        #######################           Prototype calculation process of support set           #######################
                        support_features = model(supports_HSI, supports_LiDAR)
                        if SHOT_NUM_PER_CLASS == 1:
                            support_proto = support_features
                        else:
                            support_proto = support_features.reshape(number_class, SHOT_NUM_PER_CLASS, -1).mean(dim=1)

                        proto_labels = torch.unique(support_labels)

                        #######################           Overall prototype estimate           #######################
                        tmp = support_proto.detach()
                        tmp_label = proto_labels.long()
                        proto_dict = utils.dict_storage(proto_dict, tmp, tmp_label, new_session=False)
                        all_proto = torch.stack(list(proto_dict.values()))
                        all_proto_label = torch.tensor(list(proto_dict.keys())).cuda()

                        #######################           Classification loss calculation process for support set           #######################
                        logits_support = utils.euclidean_metric(support_features, all_proto)
                        support_cls_loss = crossEntropy(logits_support, support_labels.long().cuda())

                        #######################           Classification loss calculation process for query set           #######################
                        query_features = model(querys_HSI, querys_LiDAR)
                        logits_query = utils.euclidean_metric(query_features, all_proto)
                        query_cls_loss = crossEntropy(logits_query, query_labels.long().cuda())

                        #######################           Supervised comparison loss calculation process           #######################
                        features = model(HSI_data, LiDAR_data)
                        cos_features = torch.concat((features, all_proto),dim=0).cpu()
                        combined_labels = torch.concat((label, all_proto_label), dim=0).cpu()
                        loss_con = utils.sup_constrive(cos_features.cpu(), combined_labels.cpu(), T = 0.09)

                        #######################           Model gradient backpropagation           #######################
                        loss_all = support_cls_loss + query_cls_loss + loss_con * 0.05
                        model.zero_grad()
                        loss_all.backward()
                        optimizer.step()
                        total_loss = total_loss + loss_all

                    scheduler.step()
                    total_loss = total_loss / (idx + 1)
                    print('epoch:', epoch,
                          'loss:', total_loss.data.cpu().numpy())
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

                #######################           Model storage and testing           #######################
                torch.save(state, './model/session'+ str(session) + '_model.pt')

                print("Testing session_0...")
                session0_true_cla, session0_overall_accuracy, session0_average_accuracy, session0_kappa, session0_true_label, \
                session0_test_pred, session0_cm =test_batch(model, train_loader, test_loader, proto_dict)
                print('session_0:')
                print('overall_accuracy: {0:f}'.format(session0_overall_accuracy),
                      'average_accuracy: {0:f}'.format(session0_average_accuracy),
                      'kappa:{0:f}'.format(session0_kappa))
                print("Testing session_1...")
                session1_true_cla, session1_overall_accuracy, session1_average_accuracy, session1_kappa, session1_true_label, \
                session1_test_pred, session1_cm =test_batch(model, train_loader, test_loader_session_1, proto_dict)
                print('session_1:')
                print('overall_accuracy: {0:f}'.format(session1_overall_accuracy),
                      'average_accuracy: {0:f}'.format(session1_average_accuracy),
                      'kappa:{0:f}'.format(session1_kappa))

                if session == 2:
                    print("Testing session_2...")
                    session2_true_cla, session2_overall_accuracy, session2_average_accuracy, session2_kappa, session2_true_label, \
                    session2_test_pred, session2_cm = test_batch(model, train_loader, test_loader_session_2, proto_dict)
                    print('session2:')
                    print('overall_accuracy: {0:f}'.format(session2_overall_accuracy * 100),
                          'average_accuracy: {0:f}'.format(session2_average_accuracy),
                          'kappa:{0:f}'.format(session2_kappa))

def test_batch(model, train_loader, test_loader, proto_dict):
    nclass = len(list(proto_dict.keys()))
    model.eval()
    all_proto = torch.stack(list(proto_dict.values()))
    all_proto_label = torch.tensor(list(proto_dict.keys()))
    HSI_data, LiDAR_data, label = next(train_loader.__iter__())
    HSI_data, LiDAR_data = HSI_data.cuda(), LiDAR_data.cuda()
    train_features = model(HSI_data, LiDAR_data)  # (45, 160)
    KNN_classifier = KNeighborsClassifier(n_neighbors=1)
    KNN_classifier.fit(train_features.cpu().detach().numpy(), label)  # .cpu().detach().numpy()

    pred_array = np.array([], dtype=np.int64)
    true_label = np.array([], dtype=np.int64)

    for test_HSI_data, test_LiDAR_data, test_label in test_loader:
        test_HSI_data, test_LiDAR_data = test_HSI_data.cuda(), test_LiDAR_data.cuda()
        test_features = model(test_HSI_data, test_LiDAR_data)  # (100, 160)

        # test_logits = utils.euclidean_metric(test_features, all_proto)
        # predict_labels = torch.argmax(test_logits,dim = 1).cpu().detach().numpy()
        predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
        test_label = test_label.numpy()

        pred_array = np.append(pred_array, predict_labels)
        true_label = np.append(true_label, test_label)
    confusion_matrix = metrics.confusion_matrix(true_label, pred_array,labels=all_proto_label)
    overall_accuracy = metrics.accuracy_score(true_label, pred_array)
    true_cla = np.zeros(nclass, dtype=np.int64)
    for i in range(nclass):
        true_cla[i] = confusion_matrix[i, i]
    test_num_class = np.sum(confusion_matrix, 1)
    test_num = np.sum(test_num_class)
    num1 = np.sum(confusion_matrix, 0)
    po = overall_accuracy
    pe = np.sum(test_num_class * num1) / (test_num * test_num)
    kappa = (po - pe) / (1 - pe) * 100
    true_cla = np.true_divide(true_cla, test_num_class) * 100
    average_accuracy = np.average(true_cla)
    return true_cla, overall_accuracy * 100, average_accuracy, kappa, true_label, pred_array, confusion_matrix


def main(args):
    print(args)
    args.num_of_ex = 1
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train(args)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Pre training
    parser.add_argument('--old_num_perclass', type=int, default=100)
    parser.add_argument('--new_num_perclass', type=int, default=5)

    # Base
    parser.add_argument('--SHOT_RATIO_PER_CLASS', type=float, default=0.20)
    parser.add_argument('--QUERY_RATIO_PER_CLASS', type=float, default=0.80)

    parser.add_argument('--ratio', default=0.2, type=float,
                        help='ratio of val (default: 0.2)')
    # Pre training
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--windowsize', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=301)#301
    parser.add_argument('--fine_epochs', type=int, default=201)
    parser.add_argument('--session', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)

    # Augmentation
    parser.add_argument('--scale', default=9, type=int,
                        help='the minimum scale for center crop (default: 19)')

    # MAE encoder specifics
    parser.add_argument('--encoder_dim', default=64, type=int,
                        help='feature dimension for encoder (default: 64)')
    parser.add_argument('--encoder_depth', default=4, type=int,
                        help='encoder_depth; number of blocks ')
    parser.add_argument('--encoder_num_heads', default=8, type=int,
                        help='number of heads of encoder (default: 8)')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    main(args)