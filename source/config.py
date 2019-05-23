import argparse
import os
def arg():
    parser = argparse.ArgumentParser(description='Classifiar using triplet loss.')
    parser.add_argument('--CVDs', type=str, default='7', metavar='CUDA_VISIBLE_DEVICES',
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--server', type=int, default=16, metavar='T',
                        help='which server is being used')
    parser.add_argument('--train-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/train2',
                        metavar='dir', help='path of train set.')
    parser.add_argument('--test-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/test2',
                        metavar='dir', help='path of test set.')
    parser.add_argument('--train-set-csv', type=str,
                        default='/home/zili/memory/FaceRecognition-master/data/cifar100/train.csv', metavar='file',
                        help='path of train set.csv.')
    parser.add_argument('--num-triplet', type=int, default=10000, metavar='number',
                        help='number of triplet in dataset (default: 32)')
    parser.add_argument('--amount', default=0, type=int,
                        help='amount of each class for train data')
    parser.add_argument('--batch_n_classes', default=10, type=int,
                        help='depend on your dataset')
    parser.add_argument('--batch_n_num', default=20, type=int,
                        help='depend on your dataset, number for each class per batch')
    parser.add_argument('--train-batch-size', type=int, default=96, metavar='number',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=192, metavar='number',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epoch', type=int, default=300, metavar='number',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--embedding-size', type=int, default=128, metavar='number',
                        help='embedding size of model (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
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
    parser.add_argument('--check-path', type=str, default='/home/zili/memory/FaceRecognition-master/checkpoints',
                        metavar='folder', help='Checkpoint path')
    parser.add_argument('--method', type=str, default='batchhard', metavar='R',
                        help='method of sample, batchhard, batchall, batchrandom')
    parser.add_argument('--is-pretrained', type=bool, default=False, metavar='R',
                        help='whether model is pretrained.')
    parser.add_argument('--data_augmentation', type=bool, default=False, metavar='R',
                        help='whether data_augmentation.')
    args = parser.parse_args()
    return args

def adjustedArgs(args):
    if args.server == 31:
        args.train_set  = '/share/zili/code/triplet/data/cifar100/train2'
        args.test_set   = '/share/zili/code/triplet/data/cifar100/test2'
        args.train_set_csv = '/share/zili/code/triplet/data/cifar100/train.csv'
        args.check_path = '/share/zili/code/checkpoints'
    if args.server == 16:
        args.train_set = '/data0/zili/code/data/cifar100/train'
        args.test_set = '/data0/zili/code/triplet/data/cifar100/test2'
        args.train_set_csv = '/data0/zili/code/triplet/data/cifar100/train.csv'
        args.check_path = '/data0/zili/code/checkpoints'
    if args.server == 17:
        args.train_set = '/data/jiaxin/zili/data/cifar100/train2'
        args.test_set = '/data/jiaxin/zili/data/cifar100/test'
        args.train_set_csv = '/data/jiaxin/zili/data/cifar100/train.csv'
        args.check_path = '/data/jiaxin/zili/checkpoints'
    if args.server == 15:
        args.train_set = '/home/zili/code/data/cifar100/train'
        args.test_set = '/home/zili/code/data/cifar100/test'
        args.train_set_csv = '/home/zili/code/data/cifar100/train.csv'
        args.check_path = '/home/zili/code/checkpoints'
    return args


def getArgs():
    args = arg()
    args = adjustedArgs(args)
    return args



