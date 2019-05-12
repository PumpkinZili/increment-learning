import torch
from utils import AverageMeter
from torch.autograd import Variable
import numpy as np
class trainer():
    def __init__(self, args, optimizer, scheduler, sampler_train_loader, train_loader, test_loader, model, criterion, writer, file_writer):
        self.args                 = args
        self.optimizer            = optimizer
        self.scheduler            = scheduler
        self.sampler_train_loader = sampler_train_loader
        self.train_loader         = train_loader
        self.test_loader          = test_loader
        self.model                = model
        self.criterion            = criterion
        self.writer               = writer
        self.f                    = file_writer

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
        model.eval()
        # TODO

    def extractEmbeddings(self, model, train_loader):
        model.eval()
        NearestCentroid, embeddings, labels = [], [], []
        for i, (data, target) in enumerate(train_loader):
            data = data.cuda()
            data = Variable(data)
            output = model(data)
            embeddings.extend(output.data)
            # KNeighbors.extend(output.data.cpu().numpy())
            labels.extend(target.data.cpu().numpy())

        return torch.stack(embeddings), np.array(labels)

    def extract_feature_mean(self, embeddings, targets):
        '''
        Extract features of images and return the average features of different labels
        Args:
        embeddings(tensor): features of images. size: [n, feature_dimension]
        targets(numpy): labels of images.
        Returns:
        means of classes. size: [known, feature_dimension]
        '''
        fts_means = []
        for label in sorted(set(targets)):
            condition = np.where(targets == label)[0]
            features = embeddings[condition]  # [480, feature_dimension]
            fts_means.append(torch.mean(features, dim=0, keepdim=False))

        return torch.stack(fts_means)  # [self.n_known + len(self.selected_classes), feature_dimension]
