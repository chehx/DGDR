'''
This code contains the models for all methods.
'''

from .resnet import resnet18, resnet50, resnet101
from .CBAM import CBAM, Flatten
import torch
import math
import random
import torch.nn as nn
import torchvision.models
import numpy as np
    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SoftLabelGCN(nn.Module):
    def __init__(self, cfg):
        super(SoftLabelGCN, self).__init__()
        self.cnn = eval(cfg.BACKBONE)(pretrained = True, in_channel = 3, avgpool=True)
        in_features = self.cnn._out_features
        
        self.cfg = cfg

        hidden_dim = int(in_features / 2)

        self.gcn1 = GraphConvolution(cfg.DATASET.NUM_CLASSES, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, in_features)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.)

        self.register_parameter('adj', nn.Parameter(torch.tensor(self.get_gcn_adj(), dtype=torch.float)))
        self.register_buffer('adj_mask', torch.tensor(
            [[0,0,0,0,0],
            [1,0,0,0,0],
            [1,1,0,0,0],
            [1,1,1,0,0],
            [1,1,1,1,0]],
            dtype=torch.float
        ))
            
        self.register_buffer('inp', torch.tensor(self.get_gcn_inp(), dtype=torch.float))
        self.register_buffer('diag', torch.eye(cfg.DATASET.NUM_CLASSES, dtype=torch.float))

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.cnn_linear = nn.Linear(in_features, cfg.DATASET.NUM_CLASSES)

        self.remove_gcngate = False
    
    def get_gcn_inp(self):
        inp = np.eye(self.cfg.DATASET.NUM_CLASSES)
        return inp

    def get_gcn_adj(self):
        adj = 0.2 * np.eye(self.cfg.DATASET.NUM_CLASSES, self.cfg.DATASET.NUM_CLASSES, 1) + 0.2 * np.eye(self.cfg.DATASET.NUM_CLASSES, self.cfg.DATASET.NUM_CLASSES, -1)
        return adj

    def forward(self, input):
        temp = self.adj * self.adj_mask.detach()
        temp = temp + temp.t()
        a = self.relu(temp) + self.diag.detach()

        D = torch.pow(a.sum(1).float(), -0.5)
        D = torch.diag(D)
        A = torch.matmul(torch.matmul(a, D).t(), D)

        x = self.gcn1(self.inp.detach(), A)
        x = self.leaky_relu(x)
        x = self.gcn2(x, A.detach())
        x = x.transpose(0, 1)
        # x = self.dropout(x)

        cnn_x = self.cnn(input)
        cnn_x = cnn_x.view(cnn_x.size(0), -1)

        # cnn_x = self.dropout(cnn_x)

        x = torch.matmul(cnn_x.detach(), x)
        
        cnn_x = self.cnn_linear(cnn_x)

        x = self.sigmoid(x)

        if self.remove_gcngate: x = 0

        #return x * cnn_x + cnn_x, temp, x
        return x * cnn_x + cnn_x
    
    def get_config_optim(self, lr_cnn, lr_gcn, lr_adj):
        return [{'params': self.cnn.parameters(), 'lr': lr_cnn},
                {'params': self.cnn_linear.parameters(), 'lr': lr_cnn},
                {'params': self.gcn1.parameters(), 'lr': lr_gcn},
                {'params': self.gcn2.parameters(), 'lr': lr_gcn},
                {'params': self.adj, 'lr': lr_adj},
                ]
        
class CABlock(nn.Module):
    def __init__(self, input, classes, k):
        super(CABlock, self).__init__()

        self.classes = classes
        self.k = k

        self.conv1 = nn.Conv2d(input, classes*k, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout2d(0.1)
        self.BN = nn.BatchNorm2d(classes*k)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AvgPool2d((1, 1))

    def forward(self, x):
        x1 = self.relu(self.BN(self.conv1(x)))

        x2 = self.dropout(x1)
        x2 = self.maxpool(x2).squeeze().squeeze()
        x2 = x2.view(x2.shape[0], self.classes, self.k)
        weight = torch.mean(x2, dim=-1)
        weight = weight.unsqueeze(dim=-1).unsqueeze(dim=-1)

        n,c,h,w = x1.size()
        res = x1.reshape(n, self.classes, self.k, h, w).sum(2)
        new = weight * res
        attention = new.reshape(n, 1, self.classes, h, w).sum(2)
        
        return attention * x

class CABNet(nn.Module):
    def __init__(self, cfg) -> None:
        super(CABNet, self).__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(0.0)
        
        self.model = eval(cfg.BACKBONE)(pretrained = True, in_channel = 3)
        in_features = self.model._out_features
        
        self.global_attention = nn.Sequential(
            CBAM(in_features),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_features, in_features//2))
        
        self.category_attention = CABlock(in_features//2, cfg.DATASET.NUM_CLASSES, 5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features//2, cfg.DATASET.NUM_CLASSES)

    def forward(self, x):

        feature = self.dropout(self.model(x))
        feature = self.global_attention(feature).unsqueeze(dim=2).unsqueeze(dim=3)
        feature = self.category_attention(feature)
        feature = (self.pool(feature)).squeeze()
        out= self.classifier(feature)

        return out

class MixupNet(nn.Module):
    def __init__(self, cfg) -> None:
        super(MixupNet, self).__init__()
        self.cfg = cfg
        
        self.model = eval(cfg.BACKBONE)(pretrained = True, in_channel = 3, avgpool=False)
        in_features = self.model._out_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features, cfg.DATASET.NUM_CLASSES)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1), 1, -1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        output_cls = self.classifier(x)
        return output_cls

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

class MixStyleNet(nn.Module):
    def __init__(self, cfg):
        super(MixStyleNet, self).__init__()
        filter = [64, 128, 256, 512, 512]

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_size = 512
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)
        self.classifier = nn.Linear(self.feature_size, cfg.DATASET.NUM_CLASSES)
        
        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, in_channel, out_channel, index):
        if index < 3:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return conv_block

    def forward(self, x):
        g_block1 = self.block1(x)
        g_block1 = self.mixstyle(g_block1)
        g_block2 = self.block2(g_block1)
        g_block2 = self.mixstyle(g_block2)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        feature = self.block5(g_block4)
        feature = self.avgpool(feature)
        output = self.classifier(feature.view(feature.size(0), -1))
        return output

class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FishrNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, cfg, network=None):
        super(FishrNet, self).__init__()

        if network is None:
            network = torchvision.models.resnet50(pretrained=True)
        self.network = network
        self.n_outputs = 2048

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.dropout = nn.Dropout(cfg.DROP_OUT)
        self.freeze_bn()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
