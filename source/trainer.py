import torch
from utils import AverageMeter, plot_confusion_matrix, plot_sparsity_histogram, gettriplet, printConfig
from torch.autograd import Variable
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix
import os
import sys
from sklearn import neighbors
class Trainer():
    def __init__(self, args, optimizer, scheduler, sampler_train_loader, train_loader, test_loader, model, criterion, writer, file_writer, save_path, classes):
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
        self.save_path            = save_path
        self.classes              = classes

        # Get valid labels for plotting confusion matirx
        valid_lbls = []
        for _, (_, lbls) in enumerate(self.test_loader):
            valid_lbls.extend(lbls.cpu().data.numpy() if lbls.is_cuda else lbls.data.numpy())
        self.valid_lbls = np.array(valid_lbls)

    def run(self):
        start_time = datetime.datetime.now()
        best_accy =0.
        for epoch in range(1, self.args.epoch+1):
            if self.scheduler is not None:
                self.scheduler.step()
            train_loss = self.train(epoch=epoch, model=self.model,criterion=self.criterion,
                              optimizer=self.optimizer,loader=self.sampler_train_loader)


            if epoch % 4 == 0:
                printConfig(self.args, self.f, self.optimizer)

                # Validate
                validate_start = datetime.datetime.now()

                with torch.no_grad():
                    benchmark = self.extractEmbeddings(model=self.model, train_loader=self.train_loader)
                embeddings, targets = benchmark  # from training set [n, feature_dimension]
                fts_means, labels = self.extract_feature_mean(embeddings, targets)  # [n, feature_dimension], [n]

                clf_knn = neighbors.KNeighborsClassifier(n_neighbors=self.args.vote).fit(embeddings.cpu().data.numpy(), targets)
                clf_ncm = neighbors.NearestCentroid().fit(fts_means.cpu().data.numpy(), labels)

                # Train accuracy
                train_accy, train_fts, train_lbls = self.validate(epoch=epoch,
                                                                  model=self.model,
                                                                  loader=self.train_loader,
                                                                  clf_knn=clf_knn,
                                                                  clf_ncm=clf_ncm)

                # Test accuracy
                valid_accy, pred_fts, pred_lbls = self.validate(epoch=epoch,
                                                                model=self.model,
                                                                loader=self.test_loader,
                                                                clf_knn=clf_knn,
                                                                clf_ncm=clf_ncm)

                info = 'Epoch: {}, Train_loss: {:.4f}, Train_accy(KNN, NCM): {:.4f}, {:.4f}, ' \
                       'Valid_accy(KNN, NCM): {:.4f}, {:.4f}, Consumed: {}s\n'.format(
                    epoch, train_loss, train_accy[0], train_accy[1], valid_accy[0], valid_accy[1],
                    (datetime.datetime.now() - validate_start).seconds)
                print(info)
                self.f.write(info + '\r\n')
                self.writer.add_scalar(tag='Train loss', scalar_value=train_loss, global_step=epoch)
                self.writer.add_scalar(tag='Train accy(KNN)', scalar_value=train_accy[0], global_step=epoch)
                self.writer.add_scalar(tag='Train accy(NCM)', scalar_value=train_accy[1], global_step=epoch)
                self.writer.add_scalar(tag='Valid accy(KNN)', scalar_value=valid_accy[0], global_step=epoch)
                self.writer.add_scalar(tag='Valid accy(NCM)', scalar_value=valid_accy[1], global_step=epoch)
                best_accy = max(best_accy, valid_accy[1])
                if train_accy[1] >= 0.98 and valid_accy[1] >= best_accy:  # save the best state

                    state_best = {
                        'epoch': epoch,
                        'model': self.model,
                        'state_dict': self.model.state_dict(),
                        'fts_means': fts_means,
                    }
                    torch.save(state_best,
                               os.path.join(self.save_path['path_pkl'], 'state_best.pth'))

                if epoch % 8 == 0:
                    print('Saving...\n')
                    # Confusion ##########################################################################
                    confusion = confusion_matrix(y_true=self.valid_lbls,
                                                 y_pred=pred_lbls)
                    plot_confusion_matrix(cm=confusion,
                                          classes=self.classes,
                                          save_path=os.path.join(self.save_path['path_cm'], 'cm_{}.png'.format(epoch)))


                    # train set  ##############################################################################
                    with torch.no_grad():
                        benchmark = self.getEmbeddings(model=self.model, loader=self.train_loader)
                    all_train_fts, all_train_lbls = benchmark
                    self.writer.add_embedding(global_step=epoch, mat=all_train_fts, metadata=all_train_lbls)


                    # test set ##############################################################################
                    with torch.no_grad():
                        benchmark = self.getEmbeddings(model=self.model, loader=self.test_loader)
                    all_test_fts, all_test_lbls = benchmark
                    self.writer.add_embedding(global_step=epoch+1000, mat=all_test_fts, metadata=all_test_lbls)


                    # train+test ##############################################################################
                    all_fts = torch.cat((all_train_fts,all_test_fts))
                    all_lbls = torch.cat((all_train_lbls,all_test_lbls))
                    self.writer.add_embedding(global_step=epoch+2000, mat=all_fts, metadata=all_lbls)


                    # Sparsity  ##########################################################################
                    fts_means = fts_means.cpu().data.numpy() if fts_means.is_cuda else fts_means.data.numpy()
                    save_dir = os.path.join(self.save_path['path_sparsity'], '{}'.format(epoch))
                    # makedirs(save_dir)  # make directory for histograms
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    plot_sparsity_histogram(features=fts_means,
                                            idx_to_name=self.classes,
                                            save_dir=save_dir)


                    # Save ##############################################################################
                    state_current = {
                        'phase': self.classes,
                        'epoch': epoch,
                        'model': self.model,
                        'state_dict': self.model.state_dict(),
                        'fts_means': fts_means,
                    }
                    torch.save(state_current,
                               os.path.join(self.save_path['path_pkl'], 'state_current.pth'))
            end_time = datetime.datetime.now()
            secs = (end_time - start_time).seconds
            self.f.write('Best accy: {:.4f}, Time comsumed: {}mins'.format(best_accy, int(secs / 60)))
            print('Best accy: {:.4f}, Time comsumed: {}mins'.format(best_accy, int(secs / 60)))


    def train(self, epoch, model, criterion, optimizer, loader):
        losses = AverageMeter()
        model.train()
        for step, (images, labels) in enumerate(loader):
            # print(labels)
            if torch.cuda.is_available() is True:
                images, labels = images.cuda(), labels.cuda()

            # Extract features
            embeddings = model(images)
            anchor, positive, negative = gettriplet(self.args.method, embeddings, labels)

            # Loss
            # triplet_term, sparse_term, pairwise_term, n_triplets, ap, an = criterion(embeddings, labels, model)
            # loss = triplet_term + sparse_term * 0.5 + pairwise_term * 0.5
            loss, ap, an = criterion(anchor, positive, negative)

            losses.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 4 ==0:
                print(ap,an)
            # print(ap, an)
            if (step+1) % 25 == 0:
                    # info = 'Epoch: {} Step: {}/{} | Train_loss: {:.3f} | Terms(triplet, sparse, pairwise): {:.3f}, {:.3f}, {:.3f} | n_triplets: {}'.format(
                #     epoch,step, len(loader), losses.avg, triplet_term, sparse_term, pairwise_term, n_triplets)
                info = 'Epoch: {} Step: {}/{} | Train_loss: {:.3f}'.format(epoch, step, len(loader), losses.avg)
                self.f.write(info + '\r\n')
                print(info)

        return losses.avg

    def validate(self, epoch, model, loader, clf_knn, clf_ncm):
        '''
            Validate the result of model in loader via KNN and NCM
        '''
        accuracies_knn = AverageMeter()
        accuracies_ncm = AverageMeter()
        pred_fts = []
        pred_lbls = []
        # Switch to evaluation mode
        model.eval()

        for step, (images, labels) in enumerate(loader):  # int, (Tensor, Tensor)
            if torch.cuda.is_available() is True:
                images = images.cuda()
            with torch.no_grad():
                fts = model(images)  # Tensor [batch_size, feature_dimension]

            predict = clf_ncm.predict(fts.cpu().data.numpy())
            count = (torch.tensor(predict) == labels.data).sum()
            # print(count.item())
            accuracies_ncm.update(count.item(), labels.size(0))
            pred_fts.extend(fts.cpu())
            pred_lbls.extend(predict)

            predict = clf_knn.predict(fts.cpu().data.numpy())
            count = (torch.tensor(predict) == labels.data).sum()
            accuracies_knn.update(count.item(), labels.size(0))
            # for ft, lbl in zip(fts, labels):
            #     # KNN
            #     # predict = self.knn(ft=ft, embeddings=embeddings, targets=targets, k_vote=self.args.vote)
            #     predict = clf_knn.predict(ft.cpu().data.numpy().reshape(1,-1))
            #     pred_fts.append(ft.cpu())
            #     pred_lbls.append(predict)
            #     if predict == lbl.data.numpy():
            #         accuracies_knn.update(1)
            #     else:
            #         accuracies_knn.update(0)
            #
            #     # NCM
            #     # predict = self.ncm(ft=ft, means=fts_means)
            #     predict = clf_ncm.predict(ft.cpu().data.numpy().reshape(1,-1))
            #     if predict == lbl.data.numpy():
            #         accuracies_ncm.update(1)
            #     else:
            #         accuracies_ncm.update(0)

        # to numpy
        pred_fts = torch.stack(pred_fts).view(-1, self.args.embedding_size).data.numpy()
        pred_lbls = np.array(pred_lbls)

        return (accuracies_knn.avg, accuracies_ncm.avg), pred_fts, pred_lbls

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
        labels = []
        for label in sorted(set(targets)):
            labels.append(label)
            condition = np.where(targets == label)[0]
            features = embeddings[condition]  # [480, feature_dimension]
            fts_means.append(torch.mean(features, dim=0, keepdim=False))

        return torch.stack(fts_means), np.array(labels)  # [self.n_known + len(self.selected_classes), feature_dimension]

    def knn(self, ft, embeddings, targets, k_vote):
        '''
        # Calculating distances
        distances = np.sum((embeddings - ft) ** 2, axis=1)  # [size, dimension] => [size]
        # Get top votes
        sorted_idx = np.argsort(distances)
        top_k_min_idx = sorted_idx[:k_vote]
        votes = targets[top_k_min_idx]
        # Voting
        voting = np.bincount(votes)
        result = np.argmax(voting)
        '''
        if torch.cuda.is_available() is True:
            ft, embeddings = ft.cuda(), embeddings.cuda()
        distances = torch.abs(embeddings - ft).pow(2).sum(1)  # (size, dimensions) => (size)
        votes_idx = torch.topk(distances, k=k_vote, largest=False)[1]  # idx
        result = np.argmax(np.bincount(targets[votes_idx.cpu().data.numpy() if votes_idx.is_cuda else votes_idx.data.numpy()]))

        return result

    def ncm(self, ft, means):
        if torch.cuda.is_available() is True:
            ft, means = ft.cuda(), means.cuda()
        distances = torch.abs(means - ft).pow(2).sum(1)
        idx = torch.topk(distances, k=1, largest=False)[1]
        idx = idx.cpu().data.numpy() if idx.is_cuda else idx.data.numpy()

        return idx[0]

    def getEmbeddings(self, loader, model):
        fea, l = torch.zeros(0), torch.zeros(0)
        for i, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model.forward(data)
            fea = torch.cat((fea, output.data.cpu()))
            l = torch.cat((l, target.data.cpu().float()))
        return fea, l
