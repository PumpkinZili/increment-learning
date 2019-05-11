import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EmbeddingNet(nn.Module):
    """ResNet50 except the classification layer to extract feature
    """

    def __init__(self, network='cifar_resnet50', pretrained=False, embedding_len=128, gap=True, freeze_parameter=False):
        """Using ResNet50 with a linear layer (2048,128) to extract feature
        """
        super(EmbeddingNet, self).__init__()
        self.freeze_parameter = freeze_parameter
        model = self.select_network(network, pretrained, extract_feature=freeze_parameter)
        modules = list(model.children())
        self.convnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, embedding_len)
        self.fc0 = nn.Linear(25088, 2048)
        self.fc0_bn = nn.BatchNorm1d(2048)
        self.relu0 = nn.ReLU()

    def forward(self, x):
        """extract feature
        Args:
            x: [batch_size, 3, 32, 32]
        return:
            output: [batch_size, 128]
        """
        output = self.convnet(x)
        output = output.view(output.size(0), -1)

        if not self.freeze_parameter:
            if output.size(1) == 25088:
                output = self.fc0(output)
                output = self.fc0_bn(output)
                output = self.relu0(output)
            output = self.fc1(output)
            output = self.fc1_bn(output)
            output = self.relu(output)
            output = self.fc2(output)
            output = self.normalize(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def normalize(self, x):
        """normalize by L2
        Args:
            x: [b, 128]
        """
        x_l2 = torch.norm(x, dim=1, p=2, keepdim=True)
        x = x / x_l2.expand_as(x)
        return x

    def select_network(self, network, pretrained, extract_feature=False):
        """select network"""
        if network == 'resnet50':
            print('using {}'.format('resnet50'))
            model = models.resnet50(pretrained=pretrained)
            model = nn.Sequential(*(list(model.children())[:-1]))

        elif network == 'resnet152':
            print('using {}'.format('resnet152'))
            model = models.resnet152(pretrained=pretrained)
            model = nn.Sequential(*(list(model.children())[:-1]))

        elif network == 'densenet161':
            print('using {}'.format('densenet161'))
            model = models.resnet152(pretrained=pretrained)
            model = nn.Sequential(*(list(model.children())[:-1]))

        elif network == 'SE-Net':
            print('using {}'.format('SE-Net'))
            #model = models.resnet152(pretrained=pretrained)
            #model = nn.Sequential(*(list(model.children())[:-1]))
            pass

        elif network == 'vgg19_bn':
            print('using {}'.format('vgg19 feature'))
            vgg19_bn = models.vgg19_bn(pretrained=pretrained)
            model = vgg19_bn.features
            if not extract_feature:
                model = nn.Sequential(
                            *(list(model.children())),
                            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
                            )

        elif network == 'cifar_resnet50':
            print('using {}'.format('cifar_resnet50'))
            model = models.resnet50(pretrained=pretrained)
            model.conv1.stride = 1
            model = nn.Sequential(
                    *(list(model.children())[0:3]),
                    *(list(model.children())[4:-2]),
                    nn.AdaptiveAvgPool2d(1)
                    )
        else:
            sys.exit(-1)

        return model

    def freeze_model(self, model):
        """freeze parameters in model"""
        print('Frezzing model:{}')
        for param in model.parameters():
            param.requires_grad = False
        return model


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes, embedding_len=128):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.ReLU()
        self.classification = nn.Linear(embedding_len, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        output = self.classification(output)
        return output

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))
