'''
train self supervised model: image+image or image+radio
'''

import argparse
import os
import time
import sys
# import yaml

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from models.dense3D import generate_model 
from models.normalize_layer import Normalize

from data.data_load import ImageDataset, get_brats2021_infer_transform,get_brats2021_train_transform
from models.mergenet_nodrop2 import TwoResNet


import numpy as np
from sklearn import metrics
from torch.nn import functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score, roc_auc_score, roc_curve, auc



def save_parser_options_to_txt(parser, file_path):
    # options = parser.parse_args([])
    with open(file_path, 'w') as f:
        f.write('parser options:\n')
        for key, value in vars(parser).items():
            f.write(f"{key}: {value}\n")             
     
def get_feature(args, train_loader):

    model= generate_model(121,n_input_channels=args.data_domain,  low_dim=args.feat_dim,num_classes=args.classify_num,
                            self_supervised_features=True, classify=True, mode=args.model_mode)  
    if args.pre_train:
        
        ###################
        pre_train_wiight = os.path.join(args.save_folder_check,args.ckpt_epoch)
  
        ckpt = torch.load(pre_train_wiight, map_location=lambda storage, loc: storage)
        model_state = model.state_dict()           
        m1=ckpt['model']
        state_dict = {k:v for k,v in m1.items() if k in model_state.keys()}
        print("***********load model***************")
        print(os.path.join(pre_train_wiight),len(state_dict))
        model_state.update(state_dict)
        model.load_state_dict(model_state)  
        print(model.load_state_dict(model_state) )
        
        
        

    model.cuda(args.gpu_ids)
    # classifier.cuda()
    model.eval()
    # classifier.eval()
    feat_memory = []
    prob_memory = []
    prob_memory1 = []
    name_memory = []
    label_memory = [] 
    outputs_logit_list = []
    # feat_memory=torch.zeros( n_data, args.feat_dim)

    for idx, (image, data_label, data_name,index) in enumerate(train_loader):
        
     
            if torch.cuda.is_available():
                image = image.cuda(args.gpu_ids,non_blocking=True)
        
            # forward
            feat, out= model(image)
            # out = classifier(feat)
            out = F.softmax(out, dim=1)
            outputs_logit = out.argmax(dim=1)
            for ind in range(len(data_name)):
        
                feat_memory.append(feat[ind,:].cpu().detach().numpy())
                prob_memory.append(out[ind][data_label[ind]].cpu().detach().numpy())
                prob_memory1.append(out[ind][1].cpu().detach().numpy())
                name_memory.append(data_name[ind])
                label_memory.append(data_label[ind].cpu().detach().numpy())
                outputs_logit_list.append(outputs_logit[ind].cpu().detach().numpy())
                
    sorted_id = sorted(range(len(prob_memory)), key=lambda k: prob_memory[k], reverse=True)
  
 
    prob_memory = [prob_memory[i] for i in sorted_id]
    name_memory = [name_memory[i] for i in sorted_id]
    label_memory1 = [label_memory[i] for i in sorted_id]   
    
                   

    prob_memory = np.array(prob_memory)
    prob_memory1 = np.array(prob_memory1)
    label_memory1 = np.array(label_memory1)
    outputs_logit_list = np.array(outputs_logit_list)
    label_logit_list = np.array(label_memory)
    
    return prob_memory,name_memory, label_memory, prob_memory, label_logit_list, outputs_logit_list

   
       
            
def parse_option():

    parser = argparse.ArgumentParser()

    # file settings
    parser.add_argument("--data_path", type=str, default='' )    
    
    parser.add_argument("--data_shape", type=str, default=[3,16, 384, 384])
    parser.add_argument("--crop_scale", type=float, default=[3,16, 384, 384])
    
    parser.add_argument("--classify_num", type=int, default=2)
    
    # train settingscheck_mode
    parser.add_argument("--mode", type=str, default="train",
                    choices=["GBM+MATE", "image+image"])
    parser.add_argument("--train_mode", type=str, default="GBM+MATE",
                        choices=["GBM+MATE"])
    parser.add_argument('--model', type=str, default="dense121_3D",
                        choices=["dense121_3D", "resnet50"])
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument("--print_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--val_freq", type=int, default=1)
    
    parser.add_argument("--batch_size", type=int, default=8)
    
    parser.add_argument("--gpu_ids", type=int, default=1) 
    
    parser.add_argument('--pre_train', default=True)
  
    parser.add_argument("--save_folder", type=str, default='')
    parser.add_argument("--sub_save_folder", type=str, default='')
    parser.add_argument('--ckpt_epoch', default='')    #57 40

    parser.add_argument("--folder_name", type=str, default='fold5_0')  #!!!
    parser.add_argument("--check_mode", type=str, default="train",choices=["test", "train"])
    parser.add_argument('--model_mode',  default='clu')
    parser.add_argument('--data_domain',  type=int, default=3)
    
    ####################saved_models_resnet__fold5\GM01_each_fold5_0_pre_train_sort_Clustering_weight_(1_0.5)_1_0.5_0.1_resnet


    opt = parser.parse_args()
    opt.save_folder_check = os.path.join(opt.save_folder,opt.sub_save_folder)  
    opt.model_phase=opt.model_mode
    opt.txt_file_test=os.path.join("data_txt_fold5_2",opt.folder_name+"_test.txt")


    opt.save_folder_check2 = os.path.join(opt.save_folder2,opt.sub_save_folder+'_'+opt.folder_name)
    
    return opt



            
            
def get_test_loader(args,txt_file_test,test_transform,mode):

    assert args.train_mode == "GBM+MATE"
   
    val_dataset = ImageDataset(args.data_path, txt_file_test, test_transform, mode)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        
    )

    n_data = len(val_dataset)
    print('number of val samples: {}'.format(n_data))

    return val_loader, n_data



def roc_figure(label_logit_list, prob_memory, save_folder_check,roc_fig_name):
    
    fpr, tpr, thresholds = roc_curve(label_logit_list, prob_memory)
    
    roc_auc = auc(fpr, tpr)
   
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(os.path.join(save_folder_check,roc_fig_name))
    print(os.path.join(save_folder_check,roc_fig_name))

def main():
    args = parse_option()
    if args.data_domain == 3:
        val_transforms = get_brats2021_infer_transform()

        
    val_loader, _ = get_test_loader(args,args.txt_file_test, val_transforms, mode = 'val')
    print(args.txt_file_test)

  
        
    prob_memory,name_memory, label_memory,prob_memory1, label_logit_list, outputs_logit_list= get_feature(args, val_loader)


    acc = metrics.accuracy_score(label_logit_list, outputs_logit_list) 
    recall = metrics.recall_score(label_logit_list, outputs_logit_list) 
    precision =  metrics.precision_score(label_logit_list, outputs_logit_list)
    f1 = metrics.f1_score(label_logit_list, outputs_logit_list)     
    kappa = cohen_kappa_score(label_logit_list, outputs_logit_list)
    auc = roc_auc_score(label_logit_list, prob_memory1)

    print('************************************************************************************************')
    print("auc:"+str(auc))
    print("acc:"+str(acc))
    print("recall:"+str(recall))
    print("precision:"+str(precision))
    print("f1:"+str(f1))
    print("kappa:"+str(kappa))
    print(metrics.confusion_matrix(outputs_logit_list,label_logit_list))


if __name__ == '__main__':
    main()