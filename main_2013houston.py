import argparse
from datetime import datetime
from functools import partial

import torch

import random
import numpy as np
from scipy.io import savemat
from torch.utils.data import TensorDataset, DataLoader

from utils import generate_pic
from utils import utils
from utils.augment import CenterResizeCrop
from utils.data_read import readdata, get_val_num, read_test_data, get_val_num_test
from utils.hyper_dataset import HyperData
from models.model import vit_HSI_LIDAR_prompt, vit_HSI_LIDAR
from sklearn import metrics
import tqdm
from utils.suploss import ProxyPLoss


def train(args):
    for num in range(0, args.num_of_ex):
        global proto_dict_old, proto_dict
        for session in range(args.session):
            if session == 0:
                proto_dict = {}
                num_of_samples = get_val_num(session, args.ratio, args.dataset)
                train_image_HSI, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR, validation_label, nTrain_perClass, nvalid_perClass, \
                train_index, val_index, index, image, image_LiDAR, gt, s = readdata(args.type, session, args.windowsize,
                                                                                    args.old_num_perclass, num_of_samples, num, args.dataset)

                number_class = (np.max(train_label) - np.min(train_label) + 1).astype(np.int64)
                train_set_HSI = utils.meta_dataset(train_label, train_image_HSI)
                train_set_LiDAR = utils.meta_dataset(train_label, train_image_LIDAR)
                nband = train_image_HSI.shape[1]
                nband_LiDAR = train_image_LIDAR.shape[1]
                pcl = ProxyPLoss(num_classes=number_class, scale=12).cuda()

                feature_encoder = vit_HSI_LIDAR(img_size=(args.windowsize, args.windowsize), in_chans=nband,
                                                in_chans_LIDAR=nband_LiDAR, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                                hid_chans=128, hid_chans_LIDAR=128, embed_dim=args.encoder_dim,
                                                depth=args.encoder_depth,
                                                patch_size=args.patch_size,
                                                num_heads=args.encoder_num_heads, mlp_ratio=2.0,
                                                global_pool=False, number_class = number_class)
                feature_encoder.cuda()
                # optimizer,scheduler,loss
                optimizer = torch.optim.Adam(feature_encoder.parameters(), lr=args.pre_lr, weight_decay=1e-4)
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[71, 141], gamma=0.1,
                                                                 last_epoch=-1)
                crossEntropy = torch.nn.CrossEntropyLoss().cuda()

                task = utils.Task(train_set_HSI, train_set_LiDAR, number_class, args.SHOT_RATIO_PER_CLASS,
                                  args.QUERY_RATIO_PER_CLASS)
                SHOT_NUM_PER_CLASS = int(len(task.support_labels) / number_class)
                support_dataloader = utils.get_HBKC_data_loader(task,
                                                                num_per_class=SHOT_NUM_PER_CLASS,
                                                                scale=args.scale, windowsize=args.windowsize,
                                                                split="train", shuffle=False)
                QUERY_NUM_PER_CLASS = int(len(task.query_labels) / number_class)

                for epoch in tqdm.tqdm(range(args.epochs)):
                    query_dataloader = utils.get_HBKC_data_loader(task,
                                                                  num_per_class=QUERY_NUM_PER_CLASS,
                                                                  scale=args.scale, windowsize=args.windowsize,
                                                                  split="test", shuffle=True)
                    feature_encoder.train()

                    supports_HSI, supports_LiDAR, support_labels = next(support_dataloader.__iter__())  # (batchsize, chan, 11, 11)
                    supports_HSI, supports_LiDAR, support_labels = supports_HSI.cuda(), supports_LiDAR.cuda(), support_labels.cuda()
                    querys_HSI, querys_LiDAR, query_labels = next(query_dataloader.__iter__())  # (batchsize, chan, 11, 11)
                    querys_HSI, querys_LiDAR, query_labels = querys_HSI.cuda(), querys_LiDAR.cuda(), query_labels.cuda()

                    #Prototype calculation process of support set
                    support_features = feature_encoder(supports_HSI, supports_LiDAR)
                    if SHOT_NUM_PER_CLASS == 1:
                        support_proto = support_features
                    else:
                        support_proto = support_features.reshape(number_class, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
                    proto_labels = torch.unique(support_labels)

                    #Overall prototype estimate
                    tmp = support_proto.detach()
                    tmp_label = proto_labels.long()
                    proto_dict = utils.dict_storage(proto_dict, tmp, tmp_label)
                    all_proto = torch.stack(list(proto_dict.values()))
                    all_proto_label = torch.tensor(list(proto_dict.keys())).cuda()

                    #Classification loss calculation process for query set
                    query_features = feature_encoder(querys_HSI, querys_LiDAR)
                    logits_query = utils.euclidean_metric(query_features, all_proto)

                    logits_dist = np.argmax(logits_query.cpu().detach().numpy(), axis=1)
                    overall_accuracy = metrics.accuracy_score(logits_dist, query_labels.cpu().detach().numpy())

                    query_cls_loss = crossEntropy(logits_query, query_labels)

                    #Supervised comparison loss calculation process
                    cos_features = torch.concat((query_features, all_proto),dim=0).cpu()
                    combined_labels = torch.concat((query_labels, all_proto_label), dim=0).cpu()
                    # loss_con = pcl(query_features, query_labels, all_proto)

                    loss_con = utils.sup_constrive(cos_features, combined_labels, T = 0.09)
                    loss_all = query_cls_loss + loss_con

                    feature_encoder.zero_grad()
                    loss_all.backward()
                    optimizer.step()

                    scheduler.step()
                    print('epoch:', epoch,
                          'loss:', loss_all.data.cpu().numpy(),)
                    state = {'model': feature_encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

                torch.save(state, './model/2013houston/base_model.pt')

                true_cla, overall_accuracy, average_accuracy, kappa, true_label, \
                test_pred, test_index, cm = test_base\
                    (feature_encoder, session, num, args, all_proto)
                # torch.save(state, 'model/' + args.dataset + '/session/'+ str(session) + str(overall_accuracy) + 'net.pt')
            else:
                num_of_samples = get_val_num(session, args.ratio, args.dataset)
                train_image_HSI, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR, validation_label, nTrain_perClass, nvalid_perClass, \
                train_index, val_index, index, image, image_LiDAR, gt, s = readdata(args.type, session, args.windowsize,
                                                                                    args.new_num_perclass, num_of_samples, num, args.dataset)
                train_image_HSI, train_image_LIDAR, train_label = utils.augment_data(train_image_HSI, train_image_LIDAR, train_label)
                number_class = (np.max(train_label) - np.min(train_label) + 1).astype(np.int64)
                nband = train_image_HSI.shape[1]
                nband_LiDAR = train_image_LIDAR.shape[1]
                train_set_HSI = utils.meta_dataset(train_label, train_image_HSI)
                train_set_LiDAR = utils.meta_dataset(train_label, train_image_LIDAR)

                if args.augment:
                    transform_train = [CenterResizeCrop(scale_begin = args.scale, windowsize = args.windowsize)]
                    untrain_dataset = HyperData((train_image_HSI, train_image_LIDAR, train_label), transform_train)
                else:
                    untrain_dataset = TensorDataset(torch.tensor(train_image_HSI), torch.tensor(train_image_LIDAR), torch.tensor(train_label))
                train_loader = DataLoader(dataset = untrain_dataset, batch_size = args.batch_size, shuffle = False)


                model_prompt = vit_HSI_LIDAR_prompt(img_size=(args.windowsize, args.windowsize), in_chans=nband,
                                               in_chans_LIDAR=nband_LiDAR, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                               hid_chans=128, hid_chans_LIDAR=128, embed_dim=args.encoder_dim,
                                               depth=args.encoder_depth,
                                               patch_size=args.patch_size,
                                               num_heads=args.encoder_num_heads, mlp_ratio=2.0,
                                               global_pool=False, number_class = number_class)
                if session == 1:
                    prompt_dict = {}
                    prompt_dict_LiDAR = {}
                    proto_dict_old = proto_dict
                checkpoint_model = torch.load('./model/2013houston/base_model.pt', weights_only=True)['model']

                state_dict = model_prompt.state_dict()
                model_dict_all = {}
                for k, v in checkpoint_model.items():
                    if k in state_dict:
                        model_dict_all[k] = v
                state_dict.update(model_dict_all)
                model_prompt.load_state_dict(state_dict)

                for name, param in model_prompt.named_parameters():
                    # param.requires_grad = False
                    if name in [
                                'mlp_1.0.weight', 'mlp_1.0.bias',
                                'mlp_1.2.weight', 'mlp_1.2.bias',
                                'mlp_2.0.weight', 'mlp_2.0.bias',
                                'mlp_2.2.weight', 'mlp_2.2.bias',]:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                model_prompt.cuda()

                # optimizer,scheduler,loss
                optimizer = torch.optim.Adam(model_prompt.parameters(), lr=args.task_lr, weight_decay=1e-4)
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[301, 601], gamma=0.1,
                                                                 last_epoch=-1)

                # Old class model calculation prototype
                for idx, (HSI_data, LiDAR_data, label) in enumerate(train_loader):
                    HSI_data, LiDAR_data, label = HSI_data.cuda(), LiDAR_data.cuda(), label.cuda()
                    feature_encoder.eval()
                    with torch.no_grad():
                        # Prototype calculation process of support set (Old class model)
                        support_features_old_model = feature_encoder(HSI_data, LiDAR_data)
                        support_proto_old_model = support_features_old_model.reshape(number_class,
                                                                                         args.new_num_perclass * 4, -1).mean(dim=1)
                        proto_labels = torch.unique(label)

                        # Old class model - Overall prototype estimate
                        tmp = support_proto_old_model.detach()
                        tmp_label = proto_labels.long()
                        proto_dict_old = utils.dict_storage(proto_dict, tmp, tmp_label)
                        all_proto_old = torch.stack(list(proto_dict_old.values()))
                        all_proto_label = torch.tensor(list(proto_dict_old.keys())).cuda()

                        features_old = feature_encoder(HSI_data, LiDAR_data)

                task = utils.Task(train_set_HSI, train_set_LiDAR, number_class, args.Incremental_SHOT_RATIO_PER_CLASS,
                                  args.Incremental_QUERY_RATIO_PER_CLASS)
                SHOT_NUM_PER_CLASS = int(len(task.support_labels) / number_class)
                support_dataloader = utils.get_HBKC_data_loader(task,
                                                                num_per_class=SHOT_NUM_PER_CLASS,
                                                                scale=args.scale, windowsize=args.windowsize,
                                                                split="train", shuffle=False)
                QUERY_NUM_PER_CLASS = int(len(task.query_labels) / number_class)
                query_dataloader = utils.get_HBKC_data_loader(task,
                                                              num_per_class=QUERY_NUM_PER_CLASS,
                                                              scale=args.scale, windowsize=args.windowsize,
                                                              split="test", shuffle=False)
                for epoch in tqdm.tqdm(range(args.fine_epochs[session - 1])):
                    model_prompt.train()

                    supports_HSI, supports_LiDAR, support_labels = next(
                        support_dataloader.__iter__())  # (batchsize, chan, 11, 11)
                    supports_HSI, supports_LiDAR, support_labels = supports_HSI.cuda(), supports_LiDAR.cuda(), support_labels.cuda()
                    querys_HSI, querys_LiDAR, query_labels = next(
                        query_dataloader.__iter__())  # (batchsize, chan, 11, 11)
                    querys_HSI, querys_LiDAR, query_labels = querys_HSI.cuda(), querys_LiDAR.cuda(), query_labels.cuda()

                    noisy_data_HSI = torch.empty_like(querys_HSI).cuda()
                    noisy_data_LiDAR = torch.empty_like(querys_LiDAR).cuda()
                    # 创建类别标签
                    labels = query_labels.detach().clone().cuda()
                    # 设置噪声的比例，越大噪声越大
                    noise_scale = 0.01
                    # 为每个类别添加噪声
                    for i in range(number_class):
                        # 获取当前类别的数据
                        start_index = i * QUERY_NUM_PER_CLASS
                        end_index = start_index + QUERY_NUM_PER_CLASS
                        class_data_HSI = querys_HSI[start_index:end_index]
                        class_data_LiDAR = querys_LiDAR[start_index:end_index]
                        # 计算该类别数据的均值和方差
                        mean_HSI = class_data_HSI.mean()
                        std_dev_HSI = class_data_HSI.std()
                        mean_LiDAR = class_data_LiDAR.mean()
                        std_dev_LiDAR = class_data_LiDAR.std()
                        # 生成与当前类别数据相同形状的噪声
                        noise_HSI = torch.normal(mean=mean_HSI, std=std_dev_HSI * noise_scale, size=class_data_HSI.shape).cuda()
                        noise_LiDAR = torch.normal(mean=mean_LiDAR, std=std_dev_LiDAR * noise_scale, size=class_data_LiDAR.shape).cuda()
                        # 将噪声添加到当前类别数据中
                        noisy_data_HSI[start_index:end_index] = class_data_HSI + noise_HSI
                        noisy_data_LiDAR[start_index:end_index] = class_data_LiDAR + noise_LiDAR
                    # 打乱数据和标签
                    HSI = torch.concat((querys_HSI, noisy_data_HSI), dim=0)
                    LiDAR = torch.concat((querys_LiDAR, noisy_data_LiDAR), dim=0)
                    label = torch.concat((query_labels, labels), dim=0)
                    indices = torch.randperm(querys_HSI.size(0))
                    querys_HSI = HSI[indices]
                    querys_LiDAR = LiDAR[indices]
                    query_labels = label[indices]


                    #Model Prompt Parameter Learning for New Categories - HyperNetwork
                    query_features, prompt, prompt_LiDAR = model_prompt(querys_HSI, querys_LiDAR, features_old)


                    prompt_dict = utils.prompt_storage(prompt_dict, prompt.detach(), session)
                    prompt_dict_LiDAR = utils.prompt_storage(prompt_dict_LiDAR, prompt_LiDAR.detach(), session)

                    all_prompt = torch.stack(list(prompt_dict.values()))
                    all_prompt_LiDAR = torch.stack(list(prompt_dict_LiDAR.values()))
                    all_prompt_label = torch.tensor(list(prompt_dict.keys()))

                    #Prototype calculation process of support set
                    support_features, _, _  = model_prompt(supports_HSI, supports_LiDAR, features_old)
                    if SHOT_NUM_PER_CLASS == 1:
                        support_proto = support_features
                    else:
                        support_proto = support_features.reshape(number_class, SHOT_NUM_PER_CLASS, -1).mean(dim=1)

                    proto_labels = torch.unique(support_labels)

                    #Overall prototype estimate
                    tmp = support_proto.detach()
                    tmp_label = proto_labels.long()
                    proto_dict = utils.dict_storage(proto_dict, tmp, tmp_label)
                    all_proto = torch.stack(list(proto_dict.values()))
                    all_proto_label = torch.tensor(list(proto_dict.keys())).cuda()

                    #Classification loss calculation process for query set
                    # query_features = model_prompt(querys_HSI, querys_LiDAR)
                    logits_query = utils.euclidean_metric(query_features, all_proto)
                    query_cls_loss = crossEntropy(logits_query, query_labels)

                    logits_dist = np.argmax(logits_query.cpu().detach().numpy(), axis=1)
                    overall_accuracy = metrics.accuracy_score(logits_dist, query_labels.cpu().detach().numpy())
                    print('overall_accuracy: {0:f}'.format(overall_accuracy * 100))

                    #Supervised contrastive loss calculation process
                    cos_features = torch.concat((query_features, all_proto),dim=0).cpu()
                    combined_labels = torch.concat((query_labels, all_proto_label), dim=0).cpu()
                    loss_con = utils.sup_constrive(cos_features.cpu(), combined_labels.cpu(), T = 0.09)

                    loss_all = query_cls_loss +  loss_con

                    model_prompt.zero_grad()
                    loss_all.backward()
                    optimizer.step()
                    scheduler.step()

                    print('epoch:', epoch,
                          'loss:', loss_all.data.cpu().numpy())
                    state = {'model': model_prompt.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

                torch.save(state, './model/2013houston/session'+ str(session) + '_model.pt')


                true_cla, overall_accuracy, average_accuracy, kappa, true_label, \
                test_pred, test_index, cm =test\
                    (model_prompt, feature_encoder, session, num, args, all_proto, all_proto_old, all_prompt, all_prompt_LiDAR)
                # torch.save(state, 'model/' + args.dataset + '/session/'+ str(session) + str(overall_accuracy) + 'net.pt')

def test(model_prompt, feature_encoder, session, num, args, all_proto, all_proto_old, all_prompt, all_prompt_LiDAR):
    num_of_samples = get_val_num_test(session, args.ratio, args.dataset)
    train_image_HSI, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR, validation_label, nTrain_perClass, nvalid_perClass, \
    train_index, val_index, index, image, image_LIDAR, gt, s = read_test_data(args.type, session, args.windowsize,
                                                                        args.old_num_perclass, num_of_samples, num,
                                                                        args.dataset)
    print("Testing ...")
    model_prompt.eval()
    feature_encoder.eval()
    ind = index[0][nTrain_perClass[0] + nvalid_perClass[0]:, :]
    nclass = len(index)
    true_label = np.zeros(ind.shape[0], dtype=np.int32)
    for i in range(1, nclass):
        ddd = index[i][nTrain_perClass[i] + nvalid_perClass[i]:, :]
        ind = np.concatenate((ind, ddd), axis=0)
        tr_label = np.ones(ddd.shape[0], dtype=np.int32) * i
        true_label = np.concatenate((true_label, tr_label), axis=0)
    test_index = np.copy(ind)

    BATCH_SIZE = 1
    pred_array = np.zeros([ind.shape[0]], dtype=np.float32)
    n = ind.shape[0] // BATCH_SIZE
    windowsize = args.windowsize
    halfsize = int(windowsize / 2)
    image_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image.shape[2]], dtype=np.float32)
    image_LIDAR_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image_LIDAR.shape[2]], dtype=np.float32)

    sum_num_1 = 0
    sum_num_2 = 0
    for i in range(n):
        for j in range(BATCH_SIZE):
            m = ind[BATCH_SIZE * i + j, :]
            image_batch[j, :, :, :] = image[(m[0] - halfsize):(m[0] + halfsize + 1),
                                      (m[1] - halfsize):(m[1] + halfsize + 1), :]
            image_b = np.transpose(image_batch, (0, 3, 1, 2))
            image_LIDAR_batch[j, :, :, :] = image_LIDAR[(m[0] - halfsize):(m[0] + halfsize + 1),
                                            (m[1] - halfsize):(m[1] + halfsize + 1), :]
            image_LIDAR_b = np.transpose(image_LIDAR_batch, (0, 3, 1, 2))
        old_features = feature_encoder(torch.tensor(image_b).cuda(), torch.tensor(image_LIDAR_b).cuda())
        logits_old = utils.euclidean_metric(old_features, all_proto_old[0:9]).cpu().detach().numpy()
        index_old = np.argmax(logits_old, axis=1)
        vaule_old = logits_old[:,index_old].item()
        if session == 1:
            prompt = all_prompt[0, :, :]
            prompt_LiDAR = all_prompt_LiDAR[0, :, :]
            model_prompt.HSI_vision_net.set_prompt(prompt.unsqueeze(0))
            model_prompt.Lidar_vision_net.set_prompt(prompt_LiDAR.unsqueeze(0))
            test_features = model_prompt(torch.tensor(image_b).cuda(), torch.tensor(image_LIDAR_b).cuda(), old_features,
                                         False)
            logits_session1 = utils.euclidean_metric(test_features, all_proto[9:12,:]).cpu().detach().numpy()
            index_session1 = np.argmax(logits_session1, axis=1)
            vaule_session1 = logits_session1[:,index_session1].item()
            if vaule_session1 >= vaule_old:
                index_cls = index_session1 + 9
            else:
                index_cls = index_old
        elif session == 2:
            prompt_session1 = all_prompt[0, :, :]
            prompt_LiDAR_session1 = all_prompt_LiDAR[0, :, :]
            model_prompt.HSI_vision_net.set_prompt(prompt_session1.unsqueeze(0))
            model_prompt.Lidar_vision_net.set_prompt(prompt_LiDAR_session1.unsqueeze(0))
            test_features_session1 = model_prompt(torch.tensor(image_b).cuda(), torch.tensor(image_LIDAR_b).cuda(), old_features,
                                         False)
            logits_session1 = utils.euclidean_metric(test_features_session1, all_proto[9:12,:]).cpu().detach().numpy()
            index_session1 = np.argmax(logits_session1, axis=1)
            vaule_session1 = logits_session1[:,index_session1].item()


            prompt_session2 = all_prompt[1, :, :]
            prompt_LiDAR_session2 = all_prompt_LiDAR[1,:,:]
            model_prompt.HSI_vision_net.set_prompt(prompt_session2.unsqueeze(0))
            model_prompt.Lidar_vision_net.set_prompt(prompt_LiDAR_session2.unsqueeze(0))
            test_features_session2 = model_prompt(torch.tensor(image_b).cuda(), torch.tensor(image_LIDAR_b).cuda(), old_features, False)
            logits_session2 = utils.euclidean_metric(test_features_session2, all_proto[12:,:]).cpu().detach().numpy()
            index_session2 = np.argmax(logits_session2, axis=1)
            vaule_session2 = logits_session2[:,index_session2].item()

            if vaule_session1 >= vaule_old and  vaule_session1 >= vaule_session2:
                index_cls = index_session1 + 9
            elif vaule_old >= vaule_session1 and  vaule_old >= vaule_session2:
                index_cls = index_old
            elif vaule_session2 >= vaule_session1 and vaule_session2 >= vaule_old:
                index_cls = index_session2 + 12


        # logits_new = utils.euclidean_metric(old_features, all_proto)
        # logits_dist_new = np.argmax(logits_new.cpu().detach().numpy(), axis=1)
        #
        # if logits_dist_old == logits_dist_new:
        #     logits_dist = logits_dist_old
        # elif logits_dist_old != logits_dist_new:
        #     if logits_dist_new in [0,1,2,3]:
        #         logits_dist = logits_dist_new
        #     elif (logits_dist_new >= 4) & (logits_dist_new < 6):
        #         sum_num_1 += 1
        #         prompt = all_prompt[0,:,:]
        #         prompt_LiDAR = all_prompt_LiDAR[0,:,:]
        #         model_prompt.HSI_vision_net.set_prompt(prompt.unsqueeze(0))
        #         model_prompt.Lidar_vision_net.set_prompt(prompt_LiDAR.unsqueeze(0))
        #         test_features = model_prompt(torch.tensor(image_b).cuda(), torch.tensor(image_LIDAR_b).cuda(), old_features, False)
        #         logits = utils.euclidean_metric(test_features, all_proto[4:6,:])
        #         logits_dist = np.argmax(logits.cpu().detach().numpy(), axis=1) + 4
        #     elif (logits_dist_new >= 6) & (logits_dist_new < 8):
        #         sum_num_2 += 1
        #         prompt = all_prompt[1, :, :]
        #         prompt_LiDAR = all_prompt_LiDAR[1,:,:]
        #         model_prompt.HSI_vision_net.set_prompt(prompt.unsqueeze(0))
        #         model_prompt.Lidar_vision_net.set_prompt(prompt_LiDAR.unsqueeze(0))
        #         test_features = model_prompt(torch.tensor(image_b).cuda(), torch.tensor(image_LIDAR_b).cuda(), old_features, False)
        #         logits = utils.euclidean_metric(test_features, all_proto[6:,:])
        #         logits_dist = np.argmax(logits.cpu().detach().numpy(), axis=1) + 6

        # if isinstance(logits_dist, tuple):
        #     logits_dist = logits_dist[-1]
        pred_array[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = index_cls

    print('sum_num_1', sum_num_1)
    print('sum_num_2', sum_num_2)
    confusion_matrix = metrics.confusion_matrix(true_label, pred_array)
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
    print('OA',true_cla)
    print('overall_accuracy: {0:f}'.format(overall_accuracy * 100))
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('kappa:{0:f}'.format(kappa))

    classification_map = generate_pic.generate(image, gt, index, nTrain_perClass, nvalid_perClass, pred_array, halfsize)

    savemat('result/' + args.dataset  + '/session'+ str(session) + '_'+  str(overall_accuracy) + '.mat', {'map': classification_map})

    return true_cla, overall_accuracy * 100, average_accuracy, kappa, true_label, pred_array, test_index, confusion_matrix


def test_base(model, session, num, args, all_proto):

    num_of_samples = get_val_num_test(session, args.ratio, args.dataset)
    train_image_HSI, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR, validation_label, nTrain_perClass, nvalid_perClass, \
    train_index, val_index, index, image, image_LIDAR, gt, s = read_test_data(args.type, session, args.windowsize,
                                                                        args.old_num_perclass, num_of_samples, num,
                                                                        args.dataset)
    print("Testing ...")
    model.eval()

    ind = index[0][nTrain_perClass[0] + nvalid_perClass[0]:, :]
    nclass = len(index)
    true_label = np.zeros(ind.shape[0], dtype=np.int32)
    for i in range(1, nclass):
        ddd = index[i][nTrain_perClass[i] + nvalid_perClass[i]:, :]
        ind = np.concatenate((ind, ddd), axis=0)
        tr_label = np.ones(ddd.shape[0], dtype=np.int32) * i
        true_label = np.concatenate((true_label, tr_label), axis=0)
    test_index = np.copy(ind)

    BATCH_SIZE = args.batch_size
    pred_array = np.zeros([ind.shape[0]], dtype=np.float32)
    n = ind.shape[0] // BATCH_SIZE
    windowsize = args.windowsize
    halfsize = int(windowsize / 2)
    image_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image.shape[2]], dtype=np.float32)
    image_LIDAR_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image_LIDAR.shape[2]], dtype=np.float32)

    for i in range(n):
        for j in range(BATCH_SIZE):
            m = ind[BATCH_SIZE * i + j, :]
            image_batch[j, :, :, :] = image[(m[0] - halfsize):(m[0] + halfsize + 1),
                                      (m[1] - halfsize):(m[1] + halfsize + 1), :]
            image_b = np.transpose(image_batch, (0, 3, 1, 2))
            image_LIDAR_batch[j, :, :, :] = image_LIDAR[(m[0] - halfsize):(m[0] + halfsize + 1),
                                            (m[1] - halfsize):(m[1] + halfsize + 1), :]
            image_LIDAR_b = np.transpose(image_LIDAR_batch, (0, 3, 1, 2))
        test_features = model(torch.tensor(image_b).cuda(), torch.tensor(image_LIDAR_b).cuda())
        logits = utils.euclidean_metric(test_features, all_proto)
        logits_dist = np.argmax(logits.cpu().detach().numpy(), axis=1)

        if isinstance(logits_dist, tuple):
            logits_dist = logits_dist[-1]
        pred_array[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = logits_dist


    confusion_matrix = metrics.confusion_matrix(true_label, pred_array)
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
    print('OA',true_cla)
    print('overall_accuracy: {0:f}'.format(overall_accuracy * 100))
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('kappa:{0:f}'.format(kappa))


    classification_map = generate_pic.generate(image, gt, index, nTrain_perClass, nvalid_perClass, pred_array, halfsize)
    savemat('result/' + args.dataset  + '/session'+ str(session) + '_'+  str(overall_accuracy) + '.mat', {'map': classification_map})

    return true_cla, overall_accuracy * 100, average_accuracy, kappa, true_label, pred_array, test_index, confusion_matrix


def main(args):
    print(args)
    args.num_of_ex = 1000
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(1)  # 设置使用 GPU 1
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Pre training
    parser.add_argument('--old_num_perclass', type=int, default=180)
    parser.add_argument('--new_num_perclass', type=int, default=5)

    # Base
    parser.add_argument('--SHOT_RATIO_PER_CLASS', type=int, default=80)
    parser.add_argument('--QUERY_RATIO_PER_CLASS', type=int, default=100)

    # Incremental
    parser.add_argument('--Incremental_SHOT_RATIO_PER_CLASS', type=int, default=8)
    parser.add_argument('--Incremental_QUERY_RATIO_PER_CLASS', type=int, default=12)

    parser.add_argument('--ratio', default=0.1, type=float,
                        help='ratio of val (default: 0.1)')
    # Pre training
    parser.add_argument('--seed', dest='seed', default=114514, type=int,
                        help='Random seed')
    parser.add_argument('--windowsize', type=int, default=11)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=401)
    parser.add_argument('--fine_epochs', type=list, default=[121, 61])
    parser.add_argument('--session', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='2013houston')
    parser.add_argument('--pre_lr', type=float, default=1e-4)
    parser.add_argument('--task_lr', type=float, default=1e-1)
    parser.add_argument('--type', type=str, default='none')
    # Augmentation
    parser.add_argument('--augment', default=True, type=bool,
                        help='either use data augmentation or not (default: False)')
    parser.add_argument('--scale', default=9, type=int,
                        help='the minimum scale for center crop (default: 19)')

    #encoder specifics
    parser.add_argument('--encoder_dim', default=128, type=int,
                        help='feature dimension for encoder (default: 64)')
    parser.add_argument('--encoder_depth', default=4, type=int,
                        help='encoder_depth; number of blocks ')
    parser.add_argument('--encoder_num_heads', default=8, type=int,
                        help='number of heads of encoder (default: 8)')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    main(args)