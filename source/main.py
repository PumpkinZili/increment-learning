from config import getArgs
from tensorboardX import  SummaryWriter
from utils import makedir, TripletLoss, TripletLossV2, generate_all_triplet,generate_batch_hard_triplet,generate_random_triplets
from model import EmbeddingNet
import torch
from torch.optim import lr_scheduler
from dataset import SpecificDataset, SampledDataset, BatchSampler, LimitedBatchSampler
import torch.nn as nn
from torch.backends import cudnn
from trainer import Trainer
import sys
import os


def main():
    args = getArgs()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs
    now_time, f, save_path = makedir(args)
    writer = SummaryWriter()

    model = EmbeddingNet(network = args.model_name, pretrained=args.is_pretrained, embedding_len=args.embedding_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.5, last_epoch=-1)

    dataset = SpecificDataset(args, data_augmentation=args.data_augmentation)
    classes = dataset.classes
    train_dataset = SampledDataset(dataset.train_dataset, dataset.channels, args.amount)
    print('Train data has {}'.format(len(train_dataset)))

    test_dataset = dataset.test_dataset_fc
    # print(dataset.test_dataset.dataset_name)
    # te = dataset.test_dataset_fc
    # print(te.data_name)
    # test_dataset = SampledDataset(dataset.test_dataset, dataset.channels, args.amount)
    print('Validation data has {}'.format(len(test_dataset)))

    kwargs = {'num_workers': 8, 'pin_memory': False}
    batch_sampler = BatchSampler(train_dataset, n_classes=args.batch_n_classes, n_num=args.batch_n_num)
    # batch_sampler = LimitedBatchSampler(train_dataset, 10, args.batch_n_num, args.batch_n_classes)
    sampler_train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_sampler=batch_sampler, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=False, **kwargs)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    # print(args.method)
    # criterion = TripletLoss(margin=args.margin, method=args.method).cuda()
    criterion = TripletLossV2(margin=args.margin).cuda()
    trainer = Trainer(args, optimizer, scheduler, sampler_train_loader, train_loader, test_loader, model, criterion, writer, f, save_path, classes)
    trainer.run()
    f.close()
    writer.close()


if __name__ == '__main__':
    main()