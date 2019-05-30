import argparse
import os
from dataset import SpecificDataset, SampledDataset, BatchSampler
from utils import TripletLoss, TripletLossV2
import torch
from torch.optim import lr_scheduler
from model import EmbeddingNet
def arg():
    parser = argparse.ArgumentParser(description='Classifiar using triplet loss.')
    parser.add_argument('--CVDs', type=str, default='7', metavar='CUDA_VISIBLE_DEVICES',
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--server', type=int, default=16, metavar='T',
                        help='which server is being used')
    parser.add_argument('--train-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/train2',
                        metavar='dir', help='path of train set.')
    parser.add_argument('--train-set-old', type=str, default=None,
                        metavar='dir', help='path of old train set.')
    parser.add_argument('--test-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/test2',
                        metavar='dir', help='path of test set.')
    parser.add_argument('--train-set-csv', type=str,
                        default='/home/zili/memory/FaceRecognition-master/data/cifar100/train.csv', metavar='file',
                        help='path of train set.csv.')
    parser.add_argument('--num-triplet', type=int, default=10000, metavar='number',
                        help='number of triplet in dataset (default: 32)')
    parser.add_argument('--amount', default=0, type=int,
                        help='amount of each class for train data')
    parser.add_argument('--batch_n_classes', default=7, type=int,
                        help='depend on your dataset')
    parser.add_argument('--batch_n_num', default=5, type=int,
                        help='depend on your dataset, number for each class per batch')
    parser.add_argument('--train-batch-size', type=int, default=96, metavar='number',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=192, metavar='number',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epoch', type=int, default=300, metavar='number',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--embedding-size', type=int, default=128, metavar='number',
                        help='embedding size of model (default: 256)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--margin', type=float, default=1., metavar='margin',
                        help='loss margin (default: 1.0)')
    parser.add_argument('--dataset', default='cifar100_10', type=str,
                        help="MNIST, cifar10, cifar100, cifar100_10")
    parser.add_argument('--num-classes', type=int, default=10, metavar='number',
                        help='classes number of dataset')
    parser.add_argument('--step_size', default='30', type=int,
                        help='Scheduler step size for SGD')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--k', type=int, default=20, metavar='S',
                        help='how many images to be preserved (default: 20)')
    parser.add_argument('--vote', type=int, default=5, metavar='S',
                        help='vote for knn (default: 5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='number',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model_name', type=str, default='cifar_resnet50', metavar='M',
                        help='model name (default: cifar_resnet50)')
    parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                        help='Dropout probability (default: 0.2)')
    parser.add_argument('--pairwise', type=float, default=0.5, metavar='D',
                        help='weight of pairwise term in loss (default: 0.5)')
    parser.add_argument('--check-path', type=str, default='/home/zili/memory/FaceRecognition-master/checkpoints',
                        metavar='folder', help='Checkpoint path')
    parser.add_argument('--comment', type=str, default='',
                        metavar='string', help='comment for current train')
    parser.add_argument('--method', type=str, default='batchhard', metavar='R',
                        help='method of sample, batchhard, batchall, batchrandom')
    parser.add_argument('--pretrained', type=bool, default=False, metavar='R',
                        help='whether model is pretrained.')
    parser.add_argument('--increment', type=int, default=0, metavar='R',
                        help='which step in increment precess.')
    parser.add_argument('--data_augmentation', type=bool, default=False, metavar='R',
                        help='whether data_augmentation.')
    args = parser.parse_args()
    return args

def adjustedArgs(args):
    if args.server == 31:
        if args.dataset == 'cifar100_10':
            args.train_set  = '/share/zili/code/triplet/data/cifar100/train2'
            if args.increment == 1:
                args.train_set = '/share/zili/code/triplet/data/cifar100/increment'
            args.test_set   = '/share/zili/code/triplet/data/cifar100/test2'
        elif args.dataset == 'cifar10':
            args.train_set = '/share/zili/code/triplet/data/cifar10/train2'
            if args.increment == 1:
                args.train_set = '/share/zili/code/triplet/data/cifar10/increment'
            args.test_set = '/share/zili/code/triplet/data/cifar10/test2'
        elif args.dataset == 'mnist':
            args.train_set = '/share/zili/code/triplet/data/mnist/train'
            if args.increment == 1:
                args.train_set = '/share/zili/code/triplet/data/mnist/increment'
            args.test_set = '/share/zili/code/triplet/data/mnist/test'
        else:
            print(args.dataset)
            raise NotImplementedError
        args.check_path = '/share/zili/code/checkpoints'


    elif args.server == 16:
        if args.dataset == 'cifar100_10':
            args.train_set = '/data0/zili/code/data/cifar100/train'
            if args.increment == 1:
                args.train_set = '/data0/zili/code/data/cifar100/increment'
            args.test_set = '/data0/zili/code/data/cifar100/test'
        elif args.dataset == 'cifar10':
            args.train_set = '/data0/zili/code/data/cifar10/train'
            if args.increment == 1:
                args.train_set = '/data0/zili/code/data/cifar10/increment'
            args.test_set = '/data0/zili/code/data/cifar10/test'
        elif args.dataset == 'mnist':
            args.train_set = '/data0/zili/code/data/mnist/train'
            if args.increment == 1:
                args.train_set = '/data0/zili/code/data/mnist/increment'
            args.test_set = '/data0/zili/code/data/mnist/test'
        else:
            print(args.dataset)
            raise NotImplementedError
        args.check_path = '/data0/zili/code/checkpoints'


    elif args.server == 17:
        if args.dataset == 'cifar100_10':
            args.train_set = '/data/jiaxin/zili/data/cifar100/train2'
            args.test_set = '/data/jiaxin/zili/data/cifar100/test'
        elif args.dataset == 'cifar10':
            args.train_set = '/home/zili/code/data/cifar10/train'
            args.test_set = '/home/zili/code/data/cifar10/test'
        elif args.dataset == 'mnist':
            args.train_set = '/home/zili/code/triplet/data/mnist/train'
            args.test_set = '/home/zili/code/triplet/data/mnist/test'
        else:
            print(args.dataset)
            raise NotImplementedError
        args.check_path = '/data/jiaxin/zili/checkpoints'


    elif args.server == 15:
        if args.dataset == 'cifar100_10':
            args.train_set = '/home/zili/code/data/cifar100/train'
            if args.increment == 1:
                args.train_set = '/home/zili/code/data/cifar100/increment'
            args.test_set = '/home/zili/code/data/cifar100/test'
        elif args.dataset == 'cifar10':
            args.train_set = '/home/zili/code/data/cifar10/train'
            if args.increment == 1:
                args.train_set = '/home/zili/code/data/cifar10/increment'
            args.test_set = '/home/zili/code/data/cifar10/test'
        elif args.dataset == 'mnist':
            args.train_set = '/home/zili/code/data/mnist/train'
            if args.increment == 1:
                args.train_set = '/home/zili/code/data/mnist/increment'
            args.test_set = '/home/zili/code/data/mnist/test'
        else:
            print(args.dataset)
            raise NotImplementedError
        args.check_path = '/data0/share/zili/checkpoints'

    else:
        print(args.server,'Not server')
        raise EnvironmentError

    return args


def get_args():
    args = arg()
    args = adjustedArgs(args)
    return args


def get_model(args):
    model, preserved = None, None
    if args.increment == 0:
        model = EmbeddingNet(network = args.model_name, pretrained=args.pretrained, embedding_len=args.embedding_size)
    elif args.increment == 1:
        try:
            pkl = torch.load(args.train_set_old+'/pkl/state_best.pth')
            model = pkl['model']
            preserved = {'fts_means': pkl['fts_means'],
                'preserved_embedding': pkl['embeddings']}
        except OSError as reason:
            print(args.train_set_old+'.........')
            print(reason)
    else:
        print(args.increment)
        raise NotImplementedError

    return model, preserved


def get_osc(args, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.5, last_epoch=-1)
    # criterion = TripletLoss(margin=args.margin, method=args.method).cuda()
    criterion = TripletLossV2(margin=args.margin)

    return optimizer, scheduler, criterion


def get_dataloader(args):
    dataset = SpecificDataset(args, data_augmentation=args.data_augmentation)
    classes = dataset.classes

    train_dataset = SampledDataset(dataset.train_dataset, dataset.channels, args.amount)
    print('Train data has {}'.format(len(train_dataset)))

    test_dataset = SampledDataset(dataset.test_dataset, dataset.channels, args.amount)
    print('Validation data has {}'.format(len(test_dataset)))

    kwargs = {'num_workers': 8, 'pin_memory': False}
    batch_sampler = BatchSampler(train_dataset, n_classes=args.batch_n_classes, n_num=args.batch_n_num)
    # batch_sampler = LimitedBatchSampler(train_dataset, 10, args.batch_n_num, args.batch_n_classes)

    sampler_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    if args.increment:
        train_dataset_old = SampledDataset(dataset.train_dataset_old, dataset.channels, args.amount)
        sampler_train_loader_old = torch.utils.data.DataLoader(train_dataset_old, batch_sampler=batch_sampler, **kwargs)
        train_loader_old = torch.utils.data.DataLoader(train_dataset_old, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        train_loader_old = None
        sampler_train_loader_old = None

    return sampler_train_loader, train_loader, test_loader, sampler_train_loader_old, train_loader_old, classes
