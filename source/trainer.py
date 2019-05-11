import torch
from utils import AverageMeter
class trainer():
    def __init__(self, args, optimizer, scheduler, train_loader, test_loader, model, criterion, writer, file_writer):
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.writer = writer
        self.f = file_writer

    def run(self):
        for epoch in range(1, self.args.epoch+1):
            self.scheduler.step()

    def train(self, epoch, model, criterion, optimizer, loader):
        losses = AverageMeter()
        model.train()
        for step, (images, labels) in enumerate(loader):
            if torch.cuda.is_available() is True:
                images, labels = images.cuda(), labels.cuda()
            # Extract features
            embeddings = model(images)
            # Loss
            triplet_term, sparse_term, pairwise_term, n_triplets = criterion(embeddings, labels, model)
            loss = triplet_term + sparse_term * 0.5 + pairwise_term * 0.5
            losses.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print process
            if step % 25 == 0:
                info = 'Epoch: {} Step: {}/{} | Train_loss: {:.3f} | Terms(triplet, sparse, pairwise): {:.3f}, {:.3f}, {:.3f} | n_triplets: {}'.format(
                    epoch,step, len(loader), losses.avg, triplet_term, sparse_term, pairwise_term, n_triplets)
                self.f.writer(info + '\r\n')
                print(info)

        return losses.avg
    def validate(self, epoch, model, criterion, loader):

