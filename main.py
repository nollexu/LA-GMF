import os
import random
import time

import numpy as np

import argparse

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
import os

from model import LA_GMF

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from dataset import ADNIDataset
from utils import cal_metrics, setup_seed
from weight_loss import LogitLoss
import setproctitle

setproctitle.setproctitle('LA-GMF')


def train(epoch, optimizer, model):
    model.train()
    train_loss = 0
    total_correct = 0
    total = 0

    # start time
    start = time.perf_counter()
    for batch_idx, (data, label) in enumerate(train_loader):
        if not args.no_cuda:
            data, label = data.to(device), label.to(device)
        # can be omitted
        data, label = Variable(data), Variable(label)
        # reset gradients
        optimizer.zero_grad()
        outputs, out_f, alpha, graph_out = model(data)
        print('outputs' + str(outputs) + 'label' + str(label))
        loss_1 = criterion(outputs, label)
        loss_2 = criterion(graph_out, label)
        loss_3 = logit_loss(alpha=alpha, logits=out_f,
                            target=label.repeat(150, 1).permute(1, 0).contiguous().view(-1))
        # print('loss1', loss_1)
        # print('loss2', loss_2)
        # print('loss3', loss_3)
        if epoch <= 20:
            w = 0
        else:
            w = 0.2
        loss = loss_1 + loss_2 + (w * loss_3 / outputs.size(0))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = graph_out.max(1)
        total_correct += predicted.eq(label).sum().item()
        total += label.size(0)
    # end time
    end = time.perf_counter()
    # calculate loss for epoch
    train_loss /= len(train_loader)
    total_correct /= total
    # metrics record
    content.append(str(end - start))
    content.append(str(train_loss))
    content.append(str(total_correct))
    print('Epoch: {}, Loss: {:.4f}, Total correct: {:.4f}'.format(epoch, train_loss, total_correct))


def test(model):
    model.eval()
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    target_y = np.array([])
    pred_y = np.array([])
    start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if not args.no_cuda:
                data, label = data.to(device), label.to(device)
            data, label = Variable(data), Variable(label)
            outputs, _, _, graph_out = model(data)
            print('outputs' + str(outputs) + 'label' + str(label))
            loss = criterion(outputs, label)
            test_loss += loss.item()
            _, predicted = graph_out.max(1)
            target_y = np.append(target_y, label.tolist())
            pred_y = np.append(pred_y, predicted.tolist())
            correct += predicted.eq(label).sum().item()
            total += label.size(0)
    test_accuracy = correct / total
    test_loss /= len(test_loader)
    end = time.perf_counter()
    content.append(str(end - start))
    content.append(str(test_loss))
    content.append(str(test_accuracy))
    cf_matrix = confusion_matrix(target_y, pred_y)
    metrics_result, count_record = cal_metrics(np.array(cf_matrix))
    # precision
    content.append(str(metrics_result[0][0]))
    content.append(str(metrics_result[1][0]))
    # sensitivity
    content.append(str(metrics_result[0][1]))
    content.append(str(metrics_result[1][1]))
    # specificity
    content.append(str(metrics_result[0][2]))
    content.append(str(metrics_result[1][2]))
    # f1_score
    content.append(str(metrics_result[0][4]))
    content.append(str(metrics_result[1][4]))

    res = pd.DataFrame(columns=column)
    res.loc[0] = content
    # print('content', content)
    cur_dir = os.path.dirname(__file__)
    if os.path.exists(os.path.join(cur_dir, "cls_metrics.xlsx")):
        df = pd.read_excel(os.path.join(cur_dir, "cls_metrics.xlsx"), index_col=0)
        res = pd.concat([df, res], ignore_index=True)
        res.to_excel(os.path.join(cur_dir, "cls_metrics.xlsx"))
    else:
        res.to_excel(os.path.join(cur_dir, "cls_metrics.xlsx"))
    content.clear()

    # Save checkpoint.
    if test_accuracy > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': test_accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,
                   './checkpoint/' + 'fold' + str(fold) + '_epoch' +
                   str(epoch) + '_accuracy' + str(test_accuracy) + '.pth')
        best_acc = test_accuracy
    print('\nTest Set, Loss: {:.4f}, Test accuracy: {:.4f}'.format(test_loss, test_accuracy))


if __name__ == "__main__":
    print('Start Training')
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LA-GMF')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--reg', type=float, default=10e-4, metavar='R',
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 3)')
    parser.add_argument('--device', type=int, default=0, metavar='D',
                        help='gpu (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)
    print('current GPU', torch.cuda.current_device())
    torch.use_deterministic_algorithms(True, warn_only=True)
    setup_seed(args.seed)

    # batch_size指的是bag的数量
    data_list = []
    label_list = []
    with open(r'AD_NC_index.txt', encoding="utf-8") as file:
        content = file.readlines()
        # 逐行读取数据
        for line in content:
            data_list.append(line.split('   ')[0])
            label_list.append(line.split('   ')[1].replace('\n', ''))
    data_list = np.array(data_list)
    label_list = np.array(label_list)
    print('The total length of the dataset', len(label_list))

    # five-fold cross validation
    skf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
    fold = 0
    for train_index, test_index in skf.split(data_list, label_list):
        trainset = ADNIDataset(data_list=data_list[train_index], label_list=label_list[train_index], is_training=True)
        testset = ADNIDataset(data_list=data_list[test_index], label_list=label_list[test_index], is_training=False)
        print('fold_' + str(fold) + 'length of trainset' + str(len(trainset)))
        print('fold_' + str(fold) + 'length of testset' + str(len(testset)))

        train_loader = DataLoader(trainset, batch_size=3, shuffle=True)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)

        best_acc = 0

        print('Init Model')
        model = LA_GMF()
        if not args.no_cuda:
            model.to(device)
        criterion = nn.CrossEntropyLoss()
        logit_loss = LogitLoss()
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.reg)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
        column = (
            'train_epoch_time', 'train_loss', 'train_accuracy',
            'test_epoch_time', 'test_loss', 'test_accuracy',
            'precision_NC', 'precision_AD',
            'sensitivity_NC', 'sensitivity_AD',
            'specificity_NC', 'specificity_AD',
            'F1_NC', 'F1_AD')

        content = []

        for epoch in range(1, args.epochs + 1):
            train(epoch, optimizer, model=model)
            print('Start Testing')
            test(model)
            scheduler.step()

        fold = fold + 1
