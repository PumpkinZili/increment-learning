from config import get_args, get_model, get_dataloader, get_osc

from utils import makedir, TripletLoss, TripletLossV2
import torch
from torch.optim import lr_scheduler
from torch.backends import cudnn
from trainer import Trainer
import sys
import os


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs
    cudnn.benchmark = True


    current_time, f, save_path, writer = makedir(args)
    args.train_set_old = '/data0/zili/code/checkpoints/2019-05-26 23:25:58.449965'


    model, preserved = get_model(args)
    model = model.cuda()

    optimizer, scheduler, criterion = get_osc(args, model)
    criterion = criterion.cuda()


    sampler_train_loader, train_loader, test_loader, sampler_train_loader_old, train_loader_old, classes = get_dataloader(args)


    trainer = Trainer(args, optimizer, scheduler, sampler_train_loader, train_loader, test_loader, model,
                      preserved, sampler_train_loader_old, train_loader_old, criterion, writer, f, save_path, classes)
    trainer.run()


    f.close()
    writer.close()


if __name__ == '__main__':
    main()