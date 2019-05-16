import datetime
import os
import shutil
import random
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from os import mkdir
import itertools
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def l1_norm(vectors):
    return torch.sum(torch.abs(vectors)) / vectors.size(0)


class TripletLoss(nn.Module):
    '''
    Kernel of triplet loss.
    '''

    def __init__(self, margin, method):
        '''
        Args:
        margin (float): Margin of triplet loss
        triplet_miner: Triplet selector
        '''
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.method = method

    def forward(self, embedings, targets, model):
        # triplets, eff_rate = self.triplet_miner.get_triplets(embedings, targets, model)
        # if embedings.is_cuda:
        #     triplets = triplets.cuda()
        anchors, positives, negatives = gettriplet(self.method, embedings, targets)

        ap_distances = (anchors - positives).pow(2).sum(1)
        an_distances = (anchors - negatives).pow(2).sum(1)
        np_distances = (positives - negatives).pow(2).sum(1)
        # print()
        # Loss of triplets
        triplet_loss = F.relu(ap_distances - an_distances + self.margin)
        non_zero = torch.nonzero(triplet_loss.cpu().data).size(0)
        if non_zero == 0:
            triplet_term = triplet_loss.mean()
        else:
            triplet_term = (triplet_loss / non_zero).sum()

        sparse_term = (l1_norm(anchors) + l1_norm(positives) + l1_norm(negatives)) / 3
        pairwise_term = F.relu((ap_distances + (-an_distances) + (-np_distances)).mean())

        return triplet_term, sparse_term, pairwise_term, len(anchors), ap_distances.mean().item(), an_distances.mean().item()


class TripletLossV2(nn.Module):
    """TripletLoss and Inner class Loss together
    """

    def __init__(self, margin):
        super(TripletLossV2, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=None):
        """Average
        Args:
            size_average: None, average on semi and hard,

        """
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        triplet_loss = F.relu(distance_positive - distance_negative + self.margin)

        if size_average == None:
            return triplet_loss.mean() if size_average else triplet_loss.sum(), distance_positive.mean().item(), distance_negative.mean().item()


def gettriplet(method,embedings,targets):

    if method == 'batchhard':
        anchors, positives, negatives = generate_batch_hard_triplet(embedings, targets)
    elif method == 'batchall':
        anchors, positives, negatives = generate_all_triplet(embedings, targets)
    elif method == 'batchrandom':
        anchors, positives, negatives = generate_random_triplets(embedings, targets)
    else:
        print(method)
        raise NotImplementedError
    return anchors, positives, negatives,


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        # dist = dist - torch.diag(dist.diag)
        dist = dist - torch.diag(dist)

    return torch.clamp(dist, 0.0, np.inf)


def generate_k_triplet(embeddeds, targets, K=2, B=2):
    """
    choose close k as the positive and close B for negative
    generate N * K * B triplets
    work in cuda
    """
    batch_len = embeddeds.size(0)

    dis_mat = pairwise_distances(embeddeds).cpu().data.numpy()

    anchor, positive, negative = [], [], []

    ts = targets.reshape(-1).cpu().data.numpy()

    for i in range(batch_len):
        incls_id = np.where(ts == ts[i])[0]
        incls = dis_mat[i][incls_id]

        outcls_id = np.where(ts != ts[i])[0]
        outcls = dis_mat[i][outcls_id]

        incls_closeK = np.argsort(incls)[1:1 + K]
        outcls_closeB = np.argsort(outcls)[0:B]

        if len(incls_closeK) == 0 or len(outcls_closeB) == 0:
            continue

        an = embeddeds[i].unsqueeze(0)
        for c in incls_closeK:
            for o in outcls_closeB:
                anchor.append(an)
                positive.append(embeddeds[incls_id[c]].unsqueeze(0))
                negative.append(embeddeds[outcls_id[o]].unsqueeze(0))
    try:
        anchor = torch.cat(anchor, 0)
        positive = torch.cat(positive, 0)
        negative = torch.cat(negative, 0)
    except RuntimeError:
        print(dis_mat)
        print(anchor)
        print(positive)
        print(negative)
        print(targets)
    return anchor, positive, negative


def generate_random_triplets(embeddeds, targets, triplet_num=1000):
    """
    generate random triplets
    :param embeddeds:
    :param targets:
    :param triplet_num: number of triplets to generate
    :return:
    """
    # print("generate random triplet")
    batch_len = embeddeds.size(0)

    ts = targets.reshape(-1).cpu().data.numpy()
    anchor, positive, negative = [], [], []
    for i in range(triplet_num):
        an_id = random.randint(0, batch_len - 1)
        incls_ids = np.where(ts == ts[an_id])[0]
        while len(incls_ids) == 1:
            an_id = random.randint(0, batch_len - 1)
            incls_ids = np.where(ts == ts[an_id])[0]

        pos_id = random.choice(incls_ids)
        while pos_id == an_id:
            pos_id = random.choice(incls_ids)

        outcls_ids = np.where(ts != ts[an_id])[0]
        neg_id = random.choice(outcls_ids)

        anchor.append(embeddeds[an_id].unsqueeze(0))
        positive.append(embeddeds[pos_id].unsqueeze(0))
        negative.append(embeddeds[neg_id].unsqueeze(0))

    anchor = torch.cat(anchor, 0)
    positive = torch.cat(positive, 0)
    negative = torch.cat(negative, 0)

    return anchor, positive, negative


def generate_batch_hard_triplet(embeddeds, targets):
    """Batch Hard
    Args:
        embeddeds
        targets
    Returns:
        anchor, positive, negative
    """
    batch_len = embeddeds.size(0)

    dis_mat = pairwise_distances(embeddeds).cpu().data.numpy()
    # print(dis_mat.shape)
    anchor, positive, negative = [], [], []

    ts = targets.reshape(-1).cpu().data.numpy()
    # print(ts)
    # sys.exit(1)
    for i in range(batch_len):
        incls_id = np.nonzero(ts == ts[i])[0]  # numpy

        incls = dis_mat[i][incls_id]

        outcls_id = np.nonzero(ts != ts[i])[0]  # numpy

        outcls = dis_mat[i][outcls_id]

        if incls_id.size <= 1 or outcls_id.size < 1:
            continue

        incls_farest = np.argsort(incls)[-1]
        outcls_closest = np.argsort(outcls)[0]

        an = embeddeds[i].unsqueeze(0)
        anchor.append(an)
        positive.append(embeddeds[incls_farest].unsqueeze(0))
        negative.append(embeddeds[outcls_closest].unsqueeze(0))

    try:
        anchor = torch.cat(anchor, 0)
        positive = torch.cat(positive, 0)
        negative = torch.cat(negative, 0)
    except RuntimeError:
        print(anchor)
        print(positive)
        print(negative)

    return anchor, positive, negative


def generate_all_triplet(embeddeds, targets):
    batch_len = embeddeds.size(0)
    ts = targets.reshape(-1).cpu().data.numpy()

    un_embeddeds = embeddeds.unsqueeze(dim=1)

    anchor, positive, negative = [], [], []

    for i in range(batch_len):
        incls_id = np.nonzero(ts == ts[i])[0]

        outcls_id = np.nonzero(ts != ts[i])[0]

        if incls_id.size <= 1 or outcls_id.size < 1:
            continue

        for iid in incls_id:
            if iid == i:
                continue
            for oid in outcls_id:
                anchor.append(un_embeddeds[i])
                positive.append(un_embeddeds[iid])
                negative.append(un_embeddeds[oid])
    try:
        anchor = torch.cat(anchor, 0)
        positive = torch.cat(positive, 0)
        negative = torch.cat(negative, 0)
    except Exception as e:
        print(anchor)
        print(positive)
        print(negative)
        raise RuntimeError

    return anchor, positive, negative


class AverageMeter(object):
    '''
    Computes and stores the average.
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.cnt += n
        self.avg = self.sum / self.cnt


def makedir(args):
    now_time = str(datetime.datetime.now())
    if not os.path.exists(args.check_path):
        mkdir(args.check_path)
    args.check_path = os.path.join(args.check_path, now_time)
    if not os.path.exists(args.check_path):
        mkdir(args.check_path)
    save_path = init_path(args)
    shutil.copy('config.py', save_path['path_source'])
    shutil.copy('trainer.py', save_path['path_source'])
    shutil.copy('dataset.py', save_path['path_source'])
    shutil.copy('main.py', save_path['path_source'])
    shutil.copy('model.py', save_path['path_source'])
    shutil.copy('utils.py', save_path['path_source'])

    output1 = 'main_' + now_time
    f = open(args.check_path + os.path.sep + output1 + '.txt', 'w+')
    return now_time, f, save_path


def init_path(args):
    path_cm = os.path.join(args.check_path, 'confusion_matrix')
    path_tsne = os.path.join(args.check_path, 'tsne')
    path_tbx = os.path.join(args.check_path, 'tensorboardX')
    path_log = os.path.join(args.check_path, 'log')
    path_pkl = os.path.join(args.check_path, 'pkl')
    path_sparsity = os.path.join(args.check_path, 'sparsity')
    path_state_tsne = os.path.join(args.check_path, 'state_tsne')
    path_source = os.path.join(args.check_path, 'source')
    mkdir(path_cm)
    mkdir(path_tsne)
    mkdir(path_tbx)
    mkdir(path_log)
    mkdir(path_pkl)
    mkdir(path_sparsity)
    mkdir(path_state_tsne)
    mkdir(path_source)
    save_path = {'path_cm': path_cm, 'path_tsne': path_tsne, 'path_tbx':path_tbx, 'path_log':path_log,
                 'path_pkl':path_pkl, 'path_sparsity':path_sparsity, 'path_state_tsne':path_state_tsne, 'path_source':path_source}
    return save_path


def printConfig(args,f, optimizer):
    print("train dataset:{}".format(args.train_set))
    print("dropout: {}".format(args.dropout_p))
    print("margin:{}".format(args.margin))
    print("method: {}".format(args.method))
    print("model: {}".format(args.model_name))
    print("num_triplet: {}".format(args.num_triplet))
    print("check_path: {}".format(args.check_path))
    print("learing_rate: {}".format(args.lr))
    print("train_batch_size: {}".format(args.train_batch_size))
    print("test_batch_size: {}".format(args.test_batch_size))
    print("is_pretrained: {}".format(args.is_pretrained))
    print("optimizer: {}".format(optimizer))
    f.write("train dataset:{}".format(args.train_set) + '\r\n')
    f.write("dropout: {}".format(args.dropout_p) + '\r\n')
    f.write("margin:{}".format(args.margin) + '\r\n')
    f.write("model: {}".format(args.model_name) + '\r\n')
    f.write("method: {}".format(args.method) + '\r\n')
    f.write("num_triplet: {}".format(args.num_triplet) + '\r\n')
    f.write("check_path: {}".format(args.check_path) + '\r\n')
    f.write("learing_rate: {}".format(args.lr) + '\r\n')
    f.write("train_batch_size: {}".format(args.train_batch_size) + '\r\n')
    f.write("test_batch_size: {}".format(args.test_batch_size) + '\r\n')
    f.write("is_pretrained: {}".format(args.is_pretrained) + '\r\n')
    f.write("optimizer: {}".format(optimizer) + '\r\n')


def plot_sparsity_histogram(features, idx_to_name, save_dir):
    '''
    Args:
    features(numpy) : [len(known_classes), feature_dimensions]
    save_path(str)  : Path to save histogram
    '''
    for idx, ft in enumerate(features):
        plt.bar(range(len(ft)), ft)
        plt.savefig(os.path.join(save_dir, '{}.png'.format(idx_to_name[idx])))
        plt.close()


def plot_confusion_matrix(cm,
                          classes,
                          save_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if len(classes) <= 20:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
