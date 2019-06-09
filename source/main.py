from config import get_args, get_model, get_dataloader, get_osc

from utils import makedir, TripletLoss, TripletLossV2
import torch
from torch.optim import lr_scheduler
from torch.backends import cudnn
from trainer import Trainer
import sys
import random
import os

def main():
    args = get_args()
    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs
    cudnn.benchmark = True


    current_time, f, save_path, writer = makedir(args)
    args.train_set_old = '/data0/zili/code/checkpoints/Jun05_18-25-14'
    if args.server == 15:
        args.train_set_old = '/data0/share/zili/checkpoints/Jun09_17-21-53'


    model, preserved = get_model(args)
    model = model.cuda()

    for p in model.children():
        for c in p.parameters():
             c.requires_grad = False
        break

    optimizer, scheduler, criterion, embedding_loss = get_osc(args, model)
    criterion = criterion.cuda()
    embedding_loss = embedding_loss.cuda()


    sampler_train_loader, train_loader, test_loader, sampler_train_loader_old, train_loader_old, classes = get_dataloader(args)


    trainer = Trainer(args, optimizer, scheduler, sampler_train_loader, train_loader, test_loader, model,
                      preserved, sampler_train_loader_old, train_loader_old, criterion, embedding_loss, writer, f, save_path, classes)
    trainer.run()


    f.close()
    writer.close()


if __name__ == '__main__':
    main()