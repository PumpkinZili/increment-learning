import torch
from utils import AverageMeter, plot_confusion_matrix, plot_sparsity_histogram, gettriplet, printConfig
from torch.autograd import Variable
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix
import os
import random
import sys
import shutil
from sklearn import neighbors

class Trainer():
    def __init__(self, args, optimizer, scheduler, sampler_train_loader, train_loader, test_loader, model,
                 preserved, sampler_train_loader_old, train_loader_old, criterion, embedding_loss, writer, file_writer, save_path, classes):
        self.args                 = args
        self.optimizer            = optimizer
        self.scheduler            = scheduler
        self.sampler_train_loader = sampler_train_loader
        self.train_loader         = train_loader
        self.test_loader          = test_loader
        self.model                = model
        self.criterion            = criterion
        self.embedding_loss       = embedding_loss
        self.writer               = writer
        self.f                    = file_writer
        self.save_path            = save_path
        self.classes              = classes
        self.sampler_train_loader_old = sampler_train_loader_old
        self.train_loader_old     = train_loader_old
        self.increment_phase      = args.increment_phase
        self.means                = None
        self.preserved_embedding  = None
        if self.increment_phase > 0 :
            self.means = preserved['fts_means']
            self.preserved_embedding = preserved['preserved_embedding']
            self.preserved_embedding = self.embedding_transform(self.preserved_embedding)


        # Get valid labels for plotting confusion matrix
        valid_lbls = []
        for _, (_, lbls) in enumerate(self.test_loader):
            valid_lbls.extend(lbls.cpu().data.numpy() if lbls.is_cuda else lbls.data.numpy())
        self.valid_lbls = np.array(valid_lbls)


    def run(self):
        start_time = datetime.datetime.now()
        best_acc, best_epoch, fts_means= 0., 0, None
        for epoch in range(1, self.args.epoch+1):
            if self.scheduler is not None:
                self.scheduler.step()

            if self.increment_phase > 0:
                new_loss, ebd_loss = self.train_increment(epoch=epoch, model=self.model,criterion=self.criterion,
                                                          embedding_loss=self.embedding_loss,optimizer=self.optimizer,
                                            new_loader=self.sampler_train_loader, train_loader=self.train_loader_old)

                if epoch % 2 == 0 :
                    validate_start = datetime.datetime.now()

                    with torch.no_grad():
                        new_embeddings, new_targets = self.extractEmbeddings(self.model, self.train_loader)
                        old_embeddings, old_targets = self.extractEmbeddings(self.model, self.train_loader_old)

                    ########################################
                    embeddings = torch.cat((new_embeddings, old_embeddings))
                    targets = np.append(new_targets, old_targets)
                    fts_means, labels = self.extract_feature_mean(embeddings, targets)
                    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=self.args.vote).fit(
                        embeddings.cpu().data.numpy(), targets)
                    clf_ncm = neighbors.NearestCentroid().fit(fts_means.cpu().data.numpy(), labels)
                    # clf_ncm = neighbors.NearestCentroid().fit(self.means.cpu().data.numpy(), labels)

                    #############################################
                    # New Train accuracy
                    new_train_accy, new_train_fts, new_train_lbls = self.validate(args=self.args, model=self.model, clf_knn=clf_knn,
                                                                                  loader=self.train_loader, clf_ncm=clf_ncm)

                    # Old train acc
                    old_train_accy, old_train_fts, old_train_lbls = self.validate(args=self.args, model=self.model, clf_knn=clf_knn,
                                                                                  loader=self.train_loader_old, clf_ncm=clf_ncm)

                    # Test accuracy
                    valid_accy, pred_fts, pred_lbls = self.validate(args=self.args, model=self.model, loader=self.test_loader,
                                                                    clf_knn=clf_knn, clf_ncm=clf_ncm)

                    self.log(epoch, new_loss, new_train_accy, valid_accy, validate_start, fts_means,
                             pred_lbls, best_acc, ebd_loss, old_train_accy)

                    if (valid_accy[1] > best_acc) or epoch == self.args.epoch :
                        best_acc = max(best_acc, valid_accy[1])
                        best_epoch = epoch
                        self.save_model(epoch, fts_means, preserved_embedding=None)


            elif self.increment_phase == 0:
                train_loss = self.train_epoch(epoch=epoch, model=self.model,criterion=self.criterion,
                              optimizer=self.optimizer, new_loader=self.sampler_train_loader, pairwise=self.args.pairwise)

                if epoch % 4 == 0:
                    validate_start = datetime.datetime.now()

                    # Validate
                    with torch.no_grad():
                        embeddings, targets = self.extractEmbeddings(model=self.model, train_loader=self.train_loader)

                    fts_means, labels = self.extract_feature_mean(embeddings, targets)  # [n, feature_dimension], [n]

                    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=self.args.vote).fit(embeddings.cpu().data.numpy(), targets)
                    clf_ncm = neighbors.NearestCentroid().fit(fts_means.cpu().data.numpy(), labels)

                    # Train accuracy
                    train_accy, train_fts, train_lbls = self.validate(args=self.args, model=self.model, loader=self.train_loader,
                                                                      clf_knn=clf_knn, clf_ncm=clf_ncm)

                    # Test accuracy
                    valid_accy, pred_fts, pred_lbls = self.validate(args=self.args, model=self.model, loader=self.test_loader,
                                                                    clf_knn=clf_knn, clf_ncm=clf_ncm)


                    self.log(epoch, train_loss, train_accy, valid_accy, validate_start, fts_means,
                             pred_lbls, best_acc=best_acc)

                    if (train_accy[1] >= 0.96 and valid_accy[1] > best_acc) or epoch == self.args.epoch :
                        best_acc = max(best_acc, valid_accy[1])
                        best_epoch = epoch
                        preserved_embedding = self.preserve_image(epoch, embeddings, targets, fts_means, self.classes)
                        self.save_model(epoch, fts_means, preserved_embedding)


            elif self.increment_phase == -1:
                losses = self.train_cross_entropy(train_loader=self.train_loader, model=self.model, criterion=self.criterion,
                                        optimizer=self.optimizer, epoch=epoch)
                # TODO

            end_time = datetime.datetime.now()
            self.f.write('Best accy: {:.4f}, Best_epoch: {}, Time comsumed: {}mins'.format(best_acc, best_epoch, int(((end_time - start_time).seconds) / 60)))
            print('Best accy: {:.4f}, Best_epoch: {}, Time comsumed: {}mins'.format(best_acc, best_epoch, int(((end_time - start_time).seconds) / 60)))


    def train_increment(self, epoch, model, criterion, optimizer, new_loader, embedding_loss, train_loader=None):
        dst_losses = None
        new_losses = self.train_epoch(new_loader, model, criterion, optimizer, epoch, interval=6)
        for i in range(5):
            dst_losses = self.train_cross_entropy(train_loader, model, embedding_loss, optimizer, epoch)

            # for step, (images, labels) in enumerate(train_loader):
            #     if torch.cuda.is_available():
            #         images, labels = images.cuda(), labels.cuda()
            #     embeddings = model(images)
            #
            #     l2_loss, l1_loss = embedding_loss(embeddings, self.preserved_embedding, rm_zero=False)
            #     loss = l1_loss
            #     ebd_losses.update(loss.item())
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #     info = 'Epoch: {} Step: {}/{}| Loss: {:.3f}'.format(epoch, step + 1, len(train_loader),
            #                                                         ebd_losses.avg)
            #     self.f.write(info + '\r\n')
            #     print(info)

        return new_losses, dst_losses


    def train_epoch(self, new_loader, model, criterion, optimizer, epoch, interval=25, pairwise=0.5):
        model.train()
        losses = AverageMeter()
        for step, (images, labels) in enumerate(new_loader):

            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            embeddings = model(images)
            anchor, positive, negative, targets = gettriplet(self.args.method, embeddings, labels)

            triplet_loss, pairwise_term, center_loss, ap, an = criterion(anchor, positive, negative, targets, means=self.means,
                                                                         isSemiHard=(self.args.method == 'semihard'))
            loss = triplet_loss + pairwise_term * pairwise
            losses.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 4 == 0:
                s = str(ap) + '   ' + str(an)
                print(s)
                self.f.write(s + '\r\n')

            if (step + 1) % interval == 0:
                info = 'Epoch: {} Step: {}/{}| Train_loss: {:.3f}'.format(epoch, step + 1, len(new_loader), losses.avg)
                self.f.write(info + '\r\n')
                print(info)

        return losses.avg


    def train_cross_entropy(self, train_loader, model, criterion, optimizer, epoch):
        # model.train()
        losses = AverageMeter()
        for step, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            embeddings = model(images)

            if self.increment_phase > 0:
                l2_loss, l1_loss = criterion(embeddings, self.preserved_embedding, rm_zero=False)
                loss = l1_loss
            elif self.increment_phase == -1:
                loss = criterion(embeddings, labels)
            else:
                raise NotImplementedError
            losses.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info = 'Epoch: {} Step: {}/{}| Loss: {:.3f}'.format(epoch, step + 1, len(train_loader), losses.avg)
            self.f.write(info + '\r\n')
            print(info)
        return losses.avg


    def validate(self, args, model, loader, clf_knn, clf_ncm):
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
            with torch.no_grad():  # Tensor [batch_size, feature_dimension]
                if args.data_augmentation is True:
                    bns, ncrops, c, h, w = images.size()
                    output = model(images.view(-1, c, h, w))
                    output_avg = output.view(bns, ncrops, -1).mean(1)
                    fts = output_avg
                else:
                    fts = model(images)

            predict = clf_ncm.predict(fts.cpu().data.numpy())
            count = (torch.tensor(predict) == labels.data).sum()
            # print(count.item(), labels.size(0))
            accuracies_ncm.update(count.item(), labels.size(0))
            pred_fts.extend(fts.cpu())
            pred_lbls.extend(predict)

            predict = clf_knn.predict(fts.cpu().data.numpy())
            count = (torch.tensor(predict) == labels.data).sum()
            accuracies_knn.update(count.item(), labels.size(0))

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
            with torch.no_grad():
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


    def preserve_image(self, epoch, embeddings, targets, fts_means, classes):
        root = self.args.train_set
        image_dest = self.mk(epoch, classes)
        preserved_embedding = []

        for i, label in enumerate(sorted(set(targets))):
            condition = sorted(np.where(targets == label)[0])

            features = embeddings[condition]
            # center_image_embedding = min(features, key=lambda x: (x - fts_means[i]).pow(2).sum(0))
            # center_image_position = torch.argmax((features == center_image_embedding),dim=0,keepdim=False).item()
            # second_image_embedding = max(features, key=lambda x: (x - center_image_embedding).pow(2).sum(0))
            # second_image_position = torch.argmax((features == second_image_embedding),dim=0,keepdim=False).item()
            # third_image_embedding = max(features, key=lambda x: (x - second_image_embedding).pow(2).sum(0))
            # third_image_position = torch.argmax((features == third_image_embedding), dim=0, keepdim=False).item()

            images = sorted(random.sample(range(features.size(0)), k=self.args.k)) # select 20 images to preserve
            embedding = features[images]
            preserved_embedding.append(embedding)

            class_dir = os.path.join(root, classes[i])
            file = sorted(os.listdir(class_dir))  # get all images name

            class_dest = os.path.join(image_dest, classes[i])
            for image in images:
                image_path = os.path.join(class_dir, file[image])
                shutil.copy(image_path, class_dest)

        return torch.stack(preserved_embedding)


    def mk(self, epoch, classes):
        image_dest = self.save_path['path_images']

        if epoch == self.args.epoch:
            image_dest = os.path.join(image_dest, 'current')
        else:
            image_dest = os.path.join(image_dest, 'best')

        if not os.path.exists(image_dest): # if exist remove all, if not, make dir
            os.mkdir(image_dest)
        else:
            shutil.rmtree(image_dest)
            os.mkdir(image_dest)


        for c in classes:
            class_dest = os.path.join(image_dest, c)
            if not os.path.exists(class_dest):
                os.mkdir(class_dest)
        return image_dest


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


    def log(self, epoch, train_loss, train_accy, valid_accy, validate_start, fts_means, pred_lbls, best_acc, ebd_loss=None, old_train_accy=None):
        printConfig(self.args, self.f, self.optimizer)
        if self.increment_phase > 0:
            info = 'Epoch: {}, New_Train_loss: {:.4f}, ebd_loss: {:.4f}, New_Train_accy(KNN, NCM): ' \
                   '{:.4f}, {:.4f}, Old_Train_accy(KNN, NCM): {:.4f}, {:.4f},' \
                   'Valid_accy(KNN, NCM): {:.4f}, {:.4f}, Consumed: {}s\n'.format(
                epoch, train_loss, ebd_loss, train_accy[0], train_accy[1], old_train_accy[0],
                old_train_accy[1], valid_accy[0], valid_accy[1],
                (datetime.datetime.now() - validate_start).seconds)
        else:
            info = 'Epoch: {}, Train_loss: {:.4f}, Train_accy(KNN, NCM): {:.4f}, {:.4f}, ' \
                   'Valid_accy(KNN, NCM): {:.4f}, {:.4f}, Consumed: {}s\n'.format(
                epoch, train_loss, train_accy[0], train_accy[1], valid_accy[0], valid_accy[1],
                (datetime.datetime.now() - validate_start).seconds)
        print(info)

        self.f.write(info + '\r\n')
        self.writer.add_scalar(tag='Triplet loss', scalar_value=train_loss, global_step=epoch)
        self.writer.add_scalar(tag='Train acc(KNN)', scalar_value=train_accy[0], global_step=epoch)
        self.writer.add_scalar(tag='Train acc(NCM)', scalar_value=train_accy[1], global_step=epoch)
        self.writer.add_scalar(tag='Valid acc(KNN)', scalar_value=valid_accy[0], global_step=epoch)
        self.writer.add_scalar(tag='Valid acc(NCM)', scalar_value=valid_accy[1], global_step=epoch)
        if self.increment_phase > 0:
            self.writer.add_scalar(tag='Distillation loss', scalar_value=ebd_loss, global_step=epoch)
            self.writer.add_scalar(tag='Old Train acc(KNN)', scalar_value=old_train_accy[0], global_step=epoch)
            self.writer.add_scalar(tag='Old Train acc(NCM)', scalar_value=old_train_accy[1], global_step=epoch)

        print('Saving...\n')

        #######################################################################################

        if valid_accy[1] > best_acc or epoch == self.args.epoch:
            # Confusion ##########################################################################
            confusion = confusion_matrix(y_true=self.valid_lbls,
                                         y_pred=pred_lbls)
            plot_confusion_matrix(cm=confusion,
                                  classes=self.classes,
                                  save_path=os.path.join(self.save_path['path_cm'], 'cm_{}.png'.format(epoch)))

            # train set ##############################################################################
            with torch.no_grad():
                benchmark = self.getEmbeddings(model=self.model, loader=self.train_loader)
            all_train_fts, all_train_lbls = benchmark
            self.writer.add_embedding(global_step=epoch, mat=all_train_fts, metadata=all_train_lbls)

            # test set ##############################################################################
            with torch.no_grad():
                benchmark = self.getEmbeddings(model=self.model, loader=self.test_loader)
            all_test_fts, all_test_lbls = benchmark
            self.writer.add_embedding(global_step=epoch + 1000, mat=all_test_fts, metadata=all_test_lbls)

            # train+test ##############################################################################
            all_fts = torch.cat((all_train_fts, all_test_fts))
            all_lbls = torch.cat((all_train_lbls, all_test_lbls))
            self.writer.add_embedding(global_step=epoch + 2000, mat=all_fts, metadata=all_lbls)

            if self.increment_phase > 0:
                # Old train set ###########################################################################
                with torch.no_grad():
                    benchmark = self.getEmbeddings(model=self.model, loader=self.train_loader_old)
                all_train_fts_old, all_train_lbls_old = benchmark
                self.writer.add_embedding(global_step=epoch + 3000, mat=all_train_fts_old, metadata=all_train_lbls_old)

                # New train + Old train  ##################################################################
                new_old = torch.cat((all_train_fts, all_train_fts_old))
                new_old_lbl = torch.cat((all_train_lbls, all_train_lbls_old))
                self.writer.add_embedding(global_step=epoch + 4000, mat=new_old, metadata=new_old_lbl)

                # old train + test ########################################################################
                old_test = torch.cat((all_train_fts_old, all_test_fts))
                old_test_lbl = torch.cat((all_train_lbls_old, all_test_lbls))
                self.writer.add_embedding(global_step=epoch + 5000, mat=old_test, metadata=old_test_lbl)

            # Sparsity  ##########################################################################
            fts_means = fts_means.cpu().data.numpy() if fts_means.is_cuda else fts_means.data.numpy()
            save_dir = os.path.join(self.save_path['path_sparsity'], '{}'.format(epoch))
            # makedirs(save_dir)  # make directory for histograms
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_sparsity_histogram(features=fts_means,
                                    idx_to_name=self.classes,
                                    save_dir=save_dir)


    def save_model(self, epoch, fts_means, preserved_embedding=None):
        if epoch == self.args.epoch:
            state_best = {
                'epoch': epoch,
                'model': self.model,
                'state_dict': self.model.state_dict(),
                'fts_means': fts_means,
                'embeddings': preserved_embedding
            }
            torch.save(state_best,
                       os.path.join(self.save_path['path_pkl'], 'state_best.pth'))
        else:
            state_current = {
                'epoch': epoch,
                'model': self.model,
                'state_dict': self.model.state_dict(),
                'fts_means': fts_means,
                'embeddings': preserved_embedding
            }
            torch.save(state_current, os.path.join(self.save_path['path_pkl'], 'state_current.pth'))


    def embedding_transform(self, embedding):
        p = embedding[0]
        for i in range(1, 10):
            p = torch.cat((p, embedding[i]))
        return p