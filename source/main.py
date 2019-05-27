from config import getArgs, get_model, get_dataloader
from tensorboardX import  SummaryWriter
from utils import makedir, TripletLoss, TripletLossV2
import torch
from torch.optim import lr_scheduler
from torch.backends import cudnn
from trainer import Trainer
import sys
import os


def main():
    args = getArgs()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs
    cudnn.benchmark = True
    now_time, f, save_path = makedir(args)
    writer = SummaryWriter()

    args.train_set_old = '/data0/zili/code/checkpoints/2019-05-26 23:25:58.449965'
    model, preserved = get_model(args)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.5, last_epoch=-1)
    # criterion = TripletLoss(margin=args.margin, method=args.method).cuda()
    criterion = TripletLossV2(margin=args.margin).cuda()


    sampler_train_loader, train_loader, test_loader, sampler_train_loader_old, classes = get_dataloader(args)


    trainer = Trainer(args, optimizer, scheduler, sampler_train_loader, train_loader, test_loader, model,
                      preserved, sampler_train_loader_old, criterion, writer, f, save_path, classes)
    trainer.run()


    f.close()
    writer.close()


if __name__ == '__main__':
    main()