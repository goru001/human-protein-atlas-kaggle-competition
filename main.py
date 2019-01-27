import os
import time
import json
import torch
import random
import warnings
import torchvision
import numpy as np
import pandas as pd

from utils import *
from data import HumanDataset
from tqdm import tqdm
from config import config
from datetime import datetime
from models.models import *
from torch import nn, optim
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from class_weights import batch_weights, log_dampened_class_weights
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# 1. set random seed
random.seed(50)
np.random.seed(50)
torch.manual_seed(50)
torch.cuda.manual_seed_all(50)
# TODO: Check the warning and make sure f-score is not ill defined
warnings.filterwarnings('ignore')

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

log = Logger()
log.open("logs/%s_log_train.txt" % config.model_name, mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write(
    '                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
log.write(
    'mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
log.write(
    '-------------------------------------------------------------------------------------------------------------------------------\n')


def train(train_loader, model, criterion, optimizer, epoch, valid_loss, best_results, start):
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train()
    for i, (images, target) in enumerate(train_loader):
        # weights_for_images = np.zeros(target.shape[0])
        # for j, row in enumerate(target):
        #    weights_for_images[j] = log_dampened_class_weights[max(np.nonzero(row)).numpy()[0]]
        # weights_for_images = np.zeros(train_data_list.shape[0])
        # j = 0
        # for row in train_data_list.Target:
        #     weights_for_images[j] = max(map(lambda x: log_dampened_class_weights[int(x)], row.split(' ')))
        #     j = j + 1

        weights_for_images = torch.tensor(batch_weights[:target.shape[0]][:])
        criterion.weight = weights_for_images.type(torch.FloatTensor).cuda()


        images = images.cuda(non_blocking=True)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        # compute output
        output = model(images)

        if i == 0:
            total_output = output
            total_target = target
        else:
            total_output = torch.cat([total_output, output], 0)
            total_target = torch.cat([total_target, target], 0)


        loss = criterion(output, target)
        losses.update(loss.item(), images.size(0))

        f1_batch = f1_score(total_target.cpu(), total_output.sigmoid().cpu() > 0.15, average='macro')
        f1.update(f1_batch, images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\r', end='', flush=True)
        message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % ( \
            "train", i / len(train_loader) + epoch, epoch,
            losses.avg, f1.avg,
            valid_loss[0], valid_loss[1],
            str(best_results[0])[:8], str(best_results[1])[:8],
            time_to_str((timer() - start), 'min'))
        print(message, end='', flush=True)
    log.write("\n")
    # log.write(message)
    # log.write("\n")
    return [losses.avg, f1.avg]


# 2. evaluate function
def evaluate(val_loader, model, criterion, epoch, train_loss, best_results, start):
    # only meter loss and f1 score
    losses = AverageMeter()
    f1 = AverageMeter()
    # switch mode for evaluation
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            # image_var = Variable(images).cuda()
            # target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(images_var)

            if i==0:
                total_output = output
                total_target = target
            else:
                total_output = torch.cat([total_output, output], 0)
                total_target = torch.cat([total_target, target], 0)

            weights_for_images = torch.tensor(batch_weights[:target.shape[0]][:])
            criterion.weight = weights_for_images.type(torch.FloatTensor).cuda()

            # Discussion: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72812
            # Calculating loss on the basis of batch results
            # TODO: try with overall results like in f1 score
            loss = criterion(output, target)
            losses.update(loss.item(), images_var.size(0))

            # Calculating f1 score on the basis of overall results
            f1_batch = f1_score(total_target.cpu(), total_output.sigmoid().cpu().data.numpy() > 0.15, average='macro')
            f1.update(f1_batch, images_var.size(0))

            print('\r', end='', flush=True)
            message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % ( \
                "val", i / len(val_loader) + epoch, epoch,
                train_loss[0], train_loss[1],
                losses.avg, f1.avg,
                str(best_results[0])[:8], str(best_results[1])[:8],
                time_to_str((timer() - start), 'min'))

            print(message, end='', flush=True)
        log.write("\n")
        # log.write(message)
        # log.write("\n")
    # TODO: Shouldn't we send last f1_batch value instead of avg
    return [losses.avg, f1.avg]


# 3. test model on public dataset and save the probability matrix
def test(test_loader_512, best_models_512, folds):
    sample_submission_df = pd.read_csv("/home/gaurav/Downloads/data/protein/sample_submission.csv")
    # 3.1 confirm the model converted to cuda
    filenames, labels, submissions = [], [], []
    # for model in best_models_256:
    #     model.cuda()
    #     model.eval()
    for model in best_models_512:
        model.cuda()
        model.eval()
    submit_results = []
    # temps_256= []
    # for i, (input, filepath) in enumerate(tqdm(test_loader_256)):
    #     # 3.2 change everything to cuda and get only basename
    #     filepath = [os.path.basename(x) for x in filepath]
    #     with torch.no_grad():
    #         image_var = input.cuda(non_blocking=True)
    #         for c, model in enumerate(best_models_256):
    #             y_pred = model(image_var)
    #             if c==0:
    #                 temp = y_pred.sigmoid().cpu().data.numpy()
    #             else:
    #                 temp = temp + y_pred.sigmoid().cpu().data.numpy()
    #         temps_256.append(temp)
    #         # label = temp/config.folds
    #         # print(label > 0.5)
    #
    #         # labels.append(label > 0.15)
    #         # filenames.append(filepath)

    for i, (input, filepath) in enumerate(tqdm(test_loader_512)):
        # 3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            for c, model in enumerate(best_models_512):
                y_pred = model(image_var)
                if c==0:
                    temp = y_pred.sigmoid().cpu().data.numpy()
                else:
                    temp = temp + y_pred.sigmoid().cpu().data.numpy()
            # label = (temp + temps_256[i])/(config.folds + config.folds)
            label = temp/config.folds
            # print(label > 0.5)

            labels.append(label > 0.15)
            filenames.append(filepath)


    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/%s_bestloss_submission.csv' % config.model_name, index=None)


# 4. main function
def main():
    fold = -1
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep + str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep + str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")

    train_df = pd.read_csv("/home/gaurav/Downloads/data/protein/all_labels.csv")[:1000]
    # print(all_files)
    test_files = pd.read_csv("/home/gaurav/Downloads/data/protein/sample_submission.csv")

    # Oversampling
    # train_df_orig = train_df.copy()
    # lows = [15, 15, 15, 8, 9, 10, 8, 9, 10, 8, 9, 10, 17, 20, 24, 26, 15, 27, 15, 20, 24, 17, 8, 15, 27, 27, 27]
    # for i in lows:
    #     target = str(i)
    #     indicies = train_df_orig.loc[train_df_orig['Target'] == target].index
    #     train_df = pd.concat([train_df, train_df_orig.loc[indicies]], ignore_index=True)
    #     indicies = train_df_orig.loc[train_df_orig['Target'].str.startswith(target + " ")].index
    #     train_df = pd.concat([train_df, train_df_orig.loc[indicies]], ignore_index=True)
    #     indicies = train_df_orig.loc[train_df_orig['Target'].str.endswith(" " + target)].index
    #     train_df = pd.concat([train_df, train_df_orig.loc[indicies]], ignore_index=True)
    #     indicies = train_df_orig.loc[train_df_orig['Target'].str.contains(" " + target + " ")].index
    #     train_df = pd.concat([train_df, train_df_orig.loc[indicies]], ignore_index=True)


    # Multilabel stratified CV
    msss = MultilabelStratifiedShuffleSplit(n_splits=config.folds, test_size=0.1, random_state=0)
    train_df_orig = train_df.copy()
    X = train_df_orig['Id'].tolist()
    labels = np.array((train_df_orig['Target'].map(lambda x: [int(l) for l in x.split(' ')])).tolist())
    for c, label in enumerate(labels):
        if c==0:
            y = np.eye(config.num_classes, dtype=np.float)[label].sum(axis=0)
        else:
            y = np.vstack((y, np.eye(config.num_classes, dtype=np.float)[label].sum(axis=0)))

    for train_index, test_index in msss.split(X, y):
        fold = fold + 1
        train_df = train_df_orig.loc[train_df_orig.index.intersection(train_index)].copy()
        valid_df = train_df_orig.loc[train_df_orig.index.intersection(test_index)].copy()

    # train_df, val_data_list = train_test_split(all_files, test_size=0.2, stratify = all_files['Target'].map(lambda x: x[:3] if '27' not in x else '0'))

    # Creating weights for every image in train_data_list for weighted sampling
    # weight = [0] * len(train_data_list)
    # for idx in range(len(train_data_list)):
    #     labels = list(map(lambda x: log_dampened_class_weights[x], list(map(int, train_data_list.iloc[idx].Target.split(' ')))))
    #     weight[idx] = max(labels)
    # weight = torch.DoubleTensor(weight)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight), replacement=False)

        start_epoch = 0
        best_loss = 999
        best_f1 = 0
        best_results = [np.inf, 0]
        val_metrics = [np.inf, 0]
        resume = False

        # 4.2 get model
        model = get_net()
        model.cuda()

        # criterion
        # TODO: Change optimizer to Adam, with lr = 0.001, scheduler's step size 8 or 10, try increasing gamma to 0.5
        # Discussion: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72812
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-6, momentum=0.9)
        criterion = nn.BCEWithLogitsLoss().cuda()
        # criterion = FocalLoss().cuda()
        # criterion = F1Loss().cuda()

        train_gen = HumanDataset(train_df, config.train_data, mode="train")
        train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        val_gen = HumanDataset(valid_df, config.train_data, augument=False, mode="train")
        val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        # test_gen_256 = HumanDataset(test_files, config.test_data, augument=False, mode="test", size=256)
        test_gen_512 = HumanDataset(test_files, config.test_data, augument=False, mode="test")
        # test_loader_256 = DataLoader(test_gen_256, 1, shuffle=False, pin_memory=True, num_workers=4)
        test_loader_512 = DataLoader(test_gen_512, 1, shuffle=False, pin_memory=True, num_workers=4)

        # TODO: Try changing the scheduler
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, threshold=1e-5)
        start = timer()

        learning_rates = []

        # starting with previous best model
        best_model = torch.load(
            "%s/%s_fold_%s_model_best_loss.pth.tar" % (config.best_models, config.model_name, str(fold)))
        # best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
        model.load_state_dict(best_model["state_dict"])
        optimizer.load_state_dict(best_model['optimizer'])
        # train
        # for epoch in range(0, config.epochs):
        #     scheduler.step(epoch)
        #     # train
        #     lr = get_learning_rate(optimizer)
        #     learning_rates.append(lr)
        #
        #     # Changing sampler alternately
        #     # if epoch%2 == 0:
        #     #     train_loader = DataLoader(train_gen, batch_size=config.batch_size, sampler=sampler, shuffle=False,
        #     #                           pin_memory=True, num_workers=4)
        #     # else:
        #     #     train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True,
        #     #                               pin_memory=True, num_workers=4)
        #
        #     train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, best_results, start)
        #     # val
        #     val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, best_results, start)
        #     # check results
        #     is_best_loss = val_metrics[0] < best_results[0]
        #     best_results[0] = min(val_metrics[0], best_results[0])
        #     is_best_f1 = val_metrics[1] > best_results[1]
        #     best_results[1] = max(val_metrics[1], best_results[1])
        #     # save model
        #     save_checkpoint({
        #         "epoch": epoch + 1,
        #         "model_name": config.model_name,
        #         "state_dict": model.state_dict(),
        #         "best_loss": best_results[0],
        #         "optimizer": optimizer.state_dict(),
        #         "fold": fold,
        #         "best_f1": best_results[1],
        #     }, is_best_loss, is_best_f1, fold)
        #     # print logs
        #     print('\r', end='', flush=True)
        #     log.write(
        #         '%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % ( \
        #             "best", epoch, epoch,
        #             train_metrics[0], train_metrics[1],
        #             val_metrics[0], val_metrics[1],
        #             str(best_results[0])[:8], str(best_results[1])[:8],
        #             time_to_str((timer() - start), 'min'))
        #         )
        #     log.write("\n")
        #     time.sleep(0.01)

    # best_models_256 = []
    best_models_512 = []
    model = get_net()
    model.cuda()
    for fold in range(config.folds):
        # loading 256 sz models
        # best_model = torch.load(
        #     "%s/%s_fold_%s_model_best_loss.pth.tar" % (config.best_models_256, config.model_name, str(fold)))
        # # best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
        # model.load_state_dict(best_model["state_dict"])
        # best_models_256.append(model)
        # loading 512 sz models
        best_model = torch.load(
            "%s/%s_fold_%s_model_best_loss.pth.tar" % (config.best_models, config.model_name, str(fold)))
        model.load_state_dict(best_model["state_dict"])
        best_models_512.append(model)
        best_model = torch.load(
            "%s/%s_fold_%s_model_best_f1.pth.tar" % (config.best_models, config.model_name, str(fold)))
        model.load_state_dict(best_model["state_dict"])
        best_models_512.append(model)

    test(test_loader_512, best_models_512, fold)
    print(learning_rates)

if __name__ == "__main__":
    main()