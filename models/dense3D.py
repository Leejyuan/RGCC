import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .normalize_layer import Normalize

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.RReLU(lower=0.1, upper=0.3, inplace = True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=3,
                 conv1_t_size=3,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 low_dim = 128,
                 self_supervised_features=True,
                 classify=True,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=2,
                 mode=''):

        super().__init__()
        self.self_supervised_features = self_supervised_features
        self.classify = classify
        self.low_dim = low_dim
        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 5, 5),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 2, 2),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1',  nn.RReLU(lower=0.1, upper=0.5, inplace = True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=2, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))


        self.l2norm = Normalize(2)

        if mode == 'clu':
            self.classifier1 = nn.Sequential(
                nn.Linear(1024,1024//2),
                nn.Dropout(p=0.3),
                nn.Linear(1024//2, 1024//4),
                nn.Dropout(p=0.3)
                )

        elif mode=='ref':
            self.classifier1 = nn.Sequential(
                nn.Linear(1024,1024//2),            
                nn.Linear(1024//2, 1024//4)
                )            
            # nn.ReLU(inplace=True),)
            # ))
     
        # Linear layer
        self.classifier = nn.Linear(1024//4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        # features = F.relu(features, inplace=True)
        # features = self.self_supervised_layers(features)
        
        supervised_features = F.adaptive_avg_pool3d(features,
                                    output_size=(1,1,1)).view(features.size(0), -1)
        # supervised_features1 = features
        supervised_features = self.classifier1(supervised_features)
        # supervised_features1 = self.l2norm(features)
        out = self.classifier(supervised_features)
        
        if self.classify:
           
            return supervised_features, out
        else:
            return supervised_features
    


def generate_model(model_depth, low_dim,truth_features=False, self_supervised_features=True, classify=False,**kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24),
                         low_dim = low_dim,
                         self_supervised_features=self_supervised_features,
                         classify=classify,
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         low_dim = low_dim,
                         self_supervised_features=self_supervised_features,
                         classify=classify,
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         low_dim = low_dim,
                         self_supervised_features=self_supervised_features,
                         classify=classify,
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         low_dim = low_dim,
                         truth_features=truth_features,
                         self_supervised_features=self_supervised_features,
                         classify=classify,
                         **kwargs)

    return model


if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.Tensor(2, 2, 16, 384, 384)
    x = x.to(device)
    print("x size: {}".format(x.size()))
    
    model = generate_model(121,n_input_channels=2,num_classes=2).to(device)
    

    supervised_features, out = model(x)
    print("out size: {}".format(out.size()))