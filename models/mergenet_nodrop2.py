import torch
import torch.nn as nn
# from resnet import resnet18, resnet50, resnet101
from .dense3D import generate_model as generate_model_dense3D
from .ResNet3D import generate_model as generate_model_resnet_3D
import os
from NCE.NCEAverage_PRE import NCEAverage_PRE
import numpy as np

class TwoResNet(nn.Module):
    def __init__(self, model_name, in_channel, classify_num, mode, low_dim, pre_train,truth_features=False, self_supervised_features=True, classify=True):
        super(TwoResNet, self).__init__()
        self.mode = mode
        if model_name == 'dense121_3D':
            self.module = generate_model_dense3D(121,n_input_channels=in_channel, low_dim=low_dim, num_classes=classify_num, self_supervised_features=self_supervised_features, classify=classify)   
            if pre_train:

                ckpt = torch.load(pre_train, map_location=lambda storage, loc: storage)
                model_state = self.module.state_dict()           
                m1=ckpt['model']
                state_dict = {k:v for k,v in m1.items() if k in model_state.keys()}
                print("***********load model***************")
                print(os.path.join(pre_train),len(state_dict))
                model_state.update(state_dict)
                self.module.load_state_dict(model_state)                

        elif model_name == 'resnet18_3D':
            self.module = generate_model_resnet_3D(18,n_input_channels=in_channel,  num_classes=classify_num, self_supervised_features=self_supervised_features, classify=classify)           

        else:
            raise NotImplementedError('model {} is not implemented'.format(model_name))
  

    def forward(self, data):

  

        feat, out = self.module(data)
        
    
        return feat, out
        




    

# if __name__ == '__main__':
#     GBM_data=torch.randn(2, 3,16,384,384).cuda()
#     MATE_data=torch.randn(2, 3,16,384,384).cuda()
#     gbm_label=torch.from_numpy(np.array([0, 0]))
#     mate_label=torch.from_numpy(np.array([1, 1]))
#     idx=torch.randint(0,49,(1,2)).cuda()
#     model = TwoResNet( model_name='dense121_3D', in_channel=3, classify_num=2, low_dim=128,truth_features=False, self_supervised_features=True, classify=False).cuda()
#     ckpt = torch.load('D:\\ljy_code\Medical-Image-Classification-Contrastive-Learning\\pre_train_weight\\checkpoint_0058.pth', map_location=lambda storage, loc: storage)
#     # for k1, v in model.state_dict().items():
#     #     print(k1)
#     # for k2, v in ckpt["model"].items():
#     #     print(k2)
        
#     model_state = model.state_dict()           
#     m1=ckpt['model']
#     state_dict = {k:v for k,v in m1.items() if k in model_state.keys()}
#     model_state.update(state_dict)
#     model.load_state_dict(model_state)
#     feat_l, feat_ab = model(GBM_data, MATE_data)
#     contrast = NCEAverage(128, 49, 10, 0.07, 0.5, True).cuda()
#     a=contrast(feat_l, feat_ab,idx[0])
#     print(a)
    
    
    
    
# class HandAddResNet(nn.Module):
#     def __init__(self, fc_in_dim, model_name, in_channel, classify_num, low_dim=128):
#         super(HandAddResNet, self).__init__()
#         if model_name == 'dense3D':
#             self.radio_net = FCNet(in_dim=fc_in_dim, low_dim=low_dim, classify_num=classify_num)
#             # self.deep_net = resnet50(in_channel=in_channel, low_dim=low_dim, classify_num=classify_num)
#             self.deep_net = generate_model(121,n_input_channels=2, low_dim=low_dim, num_classes=classify_num)
#         elif model_name == 'resnet18':
#             self.radio_net = FCNet(in_dim=fc_in_dim, low_dim=low_dim, classify_num=classify_num)
#             self.deep_net = resnet18(in_channel=in_channel, low_dim=low_dim, classify_num=classify_num)
#         elif model_name == 'resnet101':
#             self.radio_net = FCNet(in_dim=fc_in_dim, low_dim=low_dim, classify_num=classify_num)
#             self.deep_net = resnet101(in_channel=in_channel, low_dim=low_dim, classify_num=classify_num)
#         else:
#             raise NotImplementedError('model {} is not implemented'.format(model_name))

#     def forward(self, image_data, radio_data, truth_features=False, self_supervised_features=False,
#                 classify=False):
#         radio_feature = self.radio_net(radio_data,
#                                        truth_features=truth_features,
#                                        self_supervised_features=self_supervised_features,
#                                        classify=classify)
#         image_feature = self.deep_net(image_data)
#         return image_feature, radio_feature