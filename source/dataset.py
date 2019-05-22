import math
import numpy as np
import torchvision
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image

class SpecificDataset(object):
    """load specific dataset
    1. dataset would be put on ../data/
    2. avaialable dataset: MNIST, cifar10, cifar100
    """

    def __init__(self, args, data_augmentation=False, iter_no=0):
        self.args = args
        self.iter_no = iter_no
        self.dataset_name = args.dataset
        self.data_augmentation = data_augmentation
        self.__load()


    def __load(self):
        if self.dataset_name == 'cifar10':
            self.n_classes = 10
            self.gap = False
            self.load_CIFAR10()
        elif self.dataset_name == 'MNIST':
            self.n_classes = 10
            self.gap = False
            self.load_MNIST()
        elif self.dataset_name == 'cifar100_10':
            self.n_classes = 10
            self.gap = True
            self.load_CIFAR100_10()
        else:
            print('Must provide valid dataset')
            sys.exit(-1)

        self.train_dataset.dataset_name = self.dataset_name
        self.test_dataset.dataset_name = self.dataset_name


    def load_MNIST(self):
        self.mean, self.std = 0.1307, 0.3081

        train_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))])

        test_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))])

        if self.args.server == 16:
            path_train = '/data0/zili/code/triplet/data/mnist/train'
            path_test = '/data0/zili/code/triplet/data/test_class'
        elif self.args.server == 15:
            path_train = '/home/zili/code/triplet/data/mnist/train'
            path_test = '/home/zili/code/triplet/data/test_class'

        self.train_dataset = torchvision.datasets.ImageFolder(path_train, transform=train_transform)
        self.train_dataset.train = True
        self.train_dataset.data, self.train_dataset.targets = self.tuple2list(self.train_dataset)

        self.test_dataset = torchvision.datasets.ImageFolder(path_test, transform=test_transform)
        self.test_dataset.data, self.test_dataset.targets = self.tuple2list(self.test_dataset)
        self.test_dataset.train = False

        self.train_dataset.train = True
        self.test_dataset.train = False

        self.width, self.height = 28, 28
        self.channels = 3

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def load_CIFAR10(self):
        self.mean = (0.49, 0.48, 0.45)
        self.std = (0.25, 0.24, 0.26)
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.mean,
                                 self.std)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean,
                                 self.std)])

        if self.args.server == 16:
            path_train = '/data0/zili/code/triplet/data/cifar'
            path_test = '/data0/zili/code/triplet/data/cifar_test'
        elif self.args.server == 15:
            raise NotImplementedError

        self.train_dataset = torchvision.datasets.ImageFolder(path_train, transform=train_transform)
        self.train_dataset.train = True
        self.train_dataset.data, self.train_dataset.targets = self.tuple2list(self.train_dataset)

        self.test_dataset = torchvision.datasets.ImageFolder(path_test, transform=test_transform)
        self.test_dataset.data, self.test_dataset.targets = self.tuple2list(self.test_dataset)
        self.test_dataset.train = False

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.width, self.height = 32, 32
        self.channels = 3



    def load_CIFAR100_10(self):
        self.mean = (0.51, 0.49, 0.44)
        self.std = (0.27, 0.26, 0.27)

        train_transform = transforms.Compose([])
        test_transform_fc = transforms.Compose([])  # use five crop
        if self.data_augmentation:
            test_transform_fc.transforms.append(transforms.Pad(4))
            test_transform_fc.transforms.append(transforms.FiveCrop(32))
            test_transform_fc.transforms.append(transforms.Lambda(lambda crops: torch.stack \
                ([transforms.Normalize(self.mean, self.std)(transforms.ToTensor()(crop)) for crop in crops])))
        else:
            test_transform_fc = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.RandomCrop((32, 32), padding=4))
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(transforms.Normalize(self.mean, self.std))
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        if self.args.server == 16:
            path_train = '/data0/zili/code/data/cifar100/train'
            path_test = '/data0/zili/code/data/cifar100/test'
        elif self.args.server == 15:
            path_train = '/home/zili/code/data/cifar100/train'
            path_test = '/home/zili/code/data/cifar100/test'
        self.train_dataset = torchvision.datasets.ImageFolder(self.args.train_set, transform=train_transform)
        self.train_dataset.train = True
        self.train_dataset.data, self.train_dataset.targets = self.tuple2list(self.train_dataset)

        self.test_dataset = torchvision.datasets.ImageFolder(self.args.test_set, transform=test_transform)
        self.test_dataset.data, self.test_dataset.targets = self.tuple2list(self.test_dataset)
        self.test_dataset.train = False

        self.test_dataset_fc = torchvision.datasets.ImageFolder(self.args.test_set, transform=test_transform_fc)
        self.test_dataset_fc.data, self.test_dataset_fc.targets = self.tuple2list(self.test_dataset_fc)
        self.test_dataset_fc.train = False
        self.test_dataset_fc.dataset_name = self.dataset_name


        self.classes = self.train_dataset.classes
        self.width, self.height = 32, 32
        self.channels = 3

    def tuple2list(self, pairs):
        data = []
        targets = []
        for img, label in pairs:
            data.append(img)
            targets.append(label)
        return data, targets


class SampledDataset(Dataset):
    """Sample data from original data"""

    def __init__(self, dataset, channels, amount):
        """
        :param dataset:
        :param channels:
        :param amount: if amount = 0, do not sample
        """
        self.train = dataset.train
        self.transform = dataset.transform
        self.channels = channels
        self.dataset_name = dataset.dataset_name
        # print(self.dataset_name)
        if self.train:
            data = dataset.data
            labels = dataset.targets
            if amount != 0:
                labels = sample_labels(labels, np.unique(labels), amount)
                data, labels = select_data_by_labels(data, labels)
            self.targets = np.array(labels)
            self.data = data
        else:
            self.targets = dataset.targets
            self.data = dataset.data

        # dict store target:imgs
        self.target_img_dict = dict()
        self.targets_uniq = list(range(len(dataset.classes)))
        for target in self.targets_uniq:
            idx = np.nonzero(self.targets == target)[0]
            self.target_img_dict.update({target: idx})

    def __getitem__(self, index):
        img = self.data[index]
        # print(path.shape)
        target = self.targets[index]
        # img = pil_loader(path)
        # img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        state = np.random.get_state()
        np.random.shuffle(self.data)
        np.random.set_state(state)
        np.random.shuffle(self.targets)



class BatchSampler(Sampler):
    """Sampler used in dataloader. Method __iter__ should output \
            the indices each time when it's called
    """

    def __init__(self, dataset, n_classes, n_num):
        super(BatchSampler, self).__init__(dataset)
        self.n_classes = n_classes
        self.n_num = n_num
        self.batch_size = self.n_classes * self.n_num
        self.targets_uniq = dataset.targets_uniq
        self.targets = np.array(dataset.targets)
        self.dataset = dataset
        self.target_img_dict = dataset.target_img_dict
        self.len = len(dataset)
        self.iter_num = len(self.targets_uniq) // self.n_classes
        self.repeat = math.ceil(self.len / self.batch_size)

    def __iter__(self):
        for _ in range(self.repeat):
            curr_p = 0
            np.random.shuffle(self.targets_uniq)
            for k, v in self.target_img_dict.items():
                np.random.shuffle(self.target_img_dict[k])

            for i in range(self.iter_num):
                target_batch = self.targets_uniq[curr_p: curr_p + self.n_classes]
                curr_p += self.n_classes
                idx = []
                for target in target_batch:
                    if len(self.target_img_dict[target]) > self.n_num:
                        idx_smp = np.random.choice(self.target_img_dict[target], self.n_num, replace=False)
                    else:
                        idx_smp = np.random.choice(self.target_img_dict[target], self.n_num, replace=True)
                    idx.extend(idx_smp.tolist())
                yield idx

    def __len__(self):
        return self.iter_num * self.repeat


class LimitedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples, n_limited):
        '''
        Args:
        dataset: Dataset
        n_classes: The number of classes totally
        n_samples: The number of samples per class
        n_limited: The number of classes per batch
        '''
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_limited = n_limited

        self.batch_size = self.n_samples * self.n_limited
        self.labels = np.array(dataset.labels)
        self.labels_set = list(set(self.labels))
        self.label_to_idx = {label: np.where(self.labels == label)[0]
                             for label in self.labels_set}

        self.iter_times = len(self.labels_set) // self.n_limited
        self.repeat = math.ceil(len(self.dataset) / (self.n_classes*self.n_samples))
        # self.repeat = math.ceil(50 / self.iter_times)  # For time saving

    def __iter__(self):
        np.random.shuffle(self.labels_set)
        for k, v in self.label_to_idx.items():
            np.random.shuffle(self.label_to_idx[k])

        for r in range(self.repeat):
            curr = 0
            for i in range(self.iter_times):
                target_classes = self.labels_set[curr: curr+self.n_limited]
                curr += self.n_limited
                indices = []
                for target in target_classes:
                    if len(self.label_to_idx[target]) > self.n_samples:
                        idx = np.random.choice(self.label_to_idx[target], self.n_samples, replace=False)
                    else:
                        idx = np.random.choice(self.label_to_idx[target], self.n_samples, replace=True)
                    indices.extend(idx.tolist())

                yield indices

    def __len__(self):
        return (self.iter_times * self.repeat)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def select_data_by_labels(data, labels):
    """select_data_by_labels
    Args:
        data: [b, c, h, w]
        labels: [b]
    Returns:
        left_data: n_classes * len(classes)
        labels: n_classes * len(n_classes)
    """
    idxs = np.nonzero(labels)
    left_data = np.take(data, idxs, axis=0)[0]
    left_labels = np.take(labels, idxs, axis=0)[0]
    left_labels = [sum(x) for x in zip(len(left_labels) * [-1], left_labels)]
    return left_data, left_labels


def sample_labels(labels, classes, amount=100):
    """labels
    Args:
        labels: a list
        classes: [0,1,2 ...]
        amount: 100
    """
    count = [0] * len(classes)
    for i in classes:
        for idx, label in enumerate(labels):
            if label == i:
                if count[i] >= amount:
                    labels[idx] = -1
                else:
                    count[i] += 1
    labels = [sum(x) for x in zip(len(labels) * [1], labels)]
    return labels
