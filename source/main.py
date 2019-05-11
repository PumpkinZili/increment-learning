from config import getArgs
from tensorboardX import  SummaryWriter
from utils import makedir, TripletLoss, generate_all_triplet,generate_batch_hard_triplet,generate_random_triplets
from model import EmbeddingNet
import torch
from torch.optim import lr_scheduler
from dataset import SpecificDataset, SampledDataset, BatchSampler
import torch.nn as nn
from torch.backends import cudnn

def main():
    args = getArgs()
    now_time, f = makedir(args)
    writer = SummaryWriter(log_dir='../runs/' + now_time)
    model = EmbeddingNet(network = args.model_name, pretrained=args.is_pretrained, embedding_len=args.embeding_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.5, last_epoch=-1)
    dataset = SpecificDataset(args, data_augmentation=False)
    n_classes = dataset.n_classes
    classes = dataset.classes
    channels = dataset.channels
    width, height = dataset.width, dataset.height
    gap = dataset.gap
    train_dataset = SampledDataset(dataset.train_dataset, channels, args.amount)
    print('Train data has {}'.format(len(train_dataset)))

    test_dataset = dataset.test_dataset
    print('Validation data has {}'.format(len(test_dataset)))
    kwargs = {'num_workers': 8, 'pin_memory': False}
    batch_sampler = BatchSampler(train_dataset, n_classes=args.batch_n_classes, n_num=args.batch_n_num)
    sampler_train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_sampler=batch_sampler, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=False, **kwargs)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    criterion = TripletLoss(margin=args.margin, method=args.method).cuda()


if __name__ == '__main__':
    main()