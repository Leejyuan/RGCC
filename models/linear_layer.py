import torch
import torch.nn as nn
import os

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, pre_train,classify_num=2):
        super(LinearClassifier, self).__init__()
        self.pre_train = pre_train
        self.classifier = nn.Linear(in_dim , classify_num)
        # self.initilize()
      
    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                # m.weight.data.normal_(0, 0.01)
                # m.bias.data.fill_(0.0)
                
    def forward(self, image_data1):
  
        output = self.classifier(image_data1)
        return output

# class Generation_LinearClassifier(nn.Module):
#     def __init__(self, in_dim, pre_train,classify_num=2):
#         super(LinearClassifier, self).__init__()
#         self.pre_train = pre_train
#         self.classifier = nn.Linear(in_dim , classify_num)
#         self.initilize()
      
#     def initilize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.constant_(m.bias, 0)
#                 # m.weight.data.normal_(0, 0.01)
#                 # m.bias.data.fill_(0.0)
                
#     def forward(self, image_data1):
  
#         output = self.classifier(image_data1)
#         return output
    
    
class HandAddLinearClassifier(nn.Module):
    def __init__(self, image_in_dim, radio_in_dim, classify_num=2):
        super(HandAddLinearClassifier, self).__init__()
        self.imageLinear = nn.Linear(image_in_dim, radio_in_dim)
        self.classifier = nn.Linear(radio_in_dim * 2, classify_num)
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, image_data, radio_data):
        image_data = self.imageLinear(image_data)
        feature = torch.cat((image_data, radio_data), dim=1)
        output = self.classifier(feature)
        return output