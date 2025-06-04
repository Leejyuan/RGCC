import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
# import nrrd
import SimpleITK as sitk
import monai.transforms as transforms
from monai.transforms.transform import MapTransform
import random
import nibabel as nib
from skimage import transform
from data.roi_expand import roi_expand2
import math

def load_cases_split2(txt_path:str,lis):
    patient_ids=[]
    labels1=[]
    labels=[]
    txtFile = open(os.path.join(txt_path),'rb')
    for line in txtFile.readlines():    
        temp = line.strip().split()         
        patient_ids.append(str(temp[0], 'utf-8') )
        if str(temp[0], 'utf-8') in lis:
            labels1.append('2')
        else:
            labels1.append(str(temp[1], 'utf-8') )
        labels.append(str(temp[1], 'utf-8') )
    return patient_ids, labels, labels1

def load_cases_split(txt_path:str):
    patient_ids=[]
    labels=[]
    txtFile = open(os.path.join(txt_path),'rb')
    for line in txtFile.readlines():    
        temp = line.strip().split()         
        patient_ids.append(str(temp[0], 'utf-8') )
        labels.append(str(temp[1], 'utf-8') )
    
    return patient_ids, labels


def nib_affine(path):
    return nib.load(path).affine


def load_cases_split_wo_label(txt_path:str):
    patient_ids=[]
    labels=[]
    txtFile = open(os.path.join(txt_path),'rb')
    for line in txtFile.readlines():    
        temp = line.strip().split()         
        patient_ids.append(str(temp[0], 'utf-8') )
     
    
    return patient_ids##, labels


def get_brats2021_base_transform():
    base_transform = [
        # [B, H, W, D] --> [B, C, H, W, D]
        transforms.EnsureChannelFirstd(keys=[ 't1', 't2', 't1ce'], channel_dim="no_channel"),      #
        transforms.Orientationd(keys=['t1', 't2', 't1ce'], axcodes="RAS"),  
        # RobustZScoreNormalization(keys=['t1', 't2','t1ce']),
        transforms.ConcatItemsd(keys=['t1', 't2', 't1ce'], name='image', dim=0),
        transforms.DeleteItemsd(keys=['t1', 't2', 't1ce']),
      
    ]
    return base_transform

def get_brats2021_base_transform_two_dom():
    base_transform = [
        # [B, H, W, D] --> [B, C, H, W, D]
        transforms.EnsureChannelFirstd(keys=[ 't1', 't2'], channel_dim="no_channel"),      #
        transforms.Orientationd(keys=['t1', 't2'], axcodes="RAS"),  
        # RobustZScoreNormalization(keys=['t1', 't2','t1ce']),
        transforms.ConcatItemsd(keys=['t1', 't2'], name='image', dim=0),
        transforms.DeleteItemsd(keys=['t1', 't2']),
      
    ]
    return base_transform


def get_brats2021_train_transform():
    base_transform = get_brats2021_base_transform()
    data_aug = [
        transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),

        # intensity aug
        transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),

    ]
    return transforms.Compose(base_transform + data_aug)


def get_brats2021_infer_transform():
    base_transform = get_brats2021_base_transform()
    
    return transforms.Compose(base_transform )

def get_brats2021_infer_transform_two_dom():
    base_transform = get_brats2021_base_transform_two_dom()
    
    return transforms.Compose(base_transform )

def read_image(file_name):

    if not os.path.exists(file_name):
        raise FileNotFoundError
    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

class ImageDataset(Dataset):
    def __init__(self, data_root, txt_file, transform, mode, self_supervised=False):
        self.data_root = data_root
        if mode == 'pre_train':
            self.class_names = load_cases_split_wo_label(txt_file)
            self.class_labels = []
            if 'gbm'in txt_file:
                for i in range(len(self.class_names)):
                    self.class_labels.append('0')
            else:
                for i in range(len(self.class_names)):
                    self.class_labels.append('1')
        elif mode == 'train':
            self.class_names ,self.class_labels = load_cases_split(txt_file)
             
        elif mode == 'val':
            self.class_names ,self.class_labels = load_cases_split(txt_file)

            # self.class_labels1 = self.class_labels
        # elif mode == 'val2':
        #     lis ,_ = load_cases_split(txt_file_010)
        #     self.class_names, self.class_labels, self.class_labels1 = load_cases_split2(txt_file,lis)
        self.transform = transform
        self.self_supervised = self_supervised
 
    def __len__(self):
        return len(self.class_names)

    def __getitem__(self, idx:int) -> tuple:
        
        data_name = self.class_names[idx]
        data_label = self.class_labels[idx]
        # data_label1 = self.class_labels1[idx]
        d=16
        w1=224
        h=224 
        data_path = os.path.join(self.data_root, data_name, data_name)
   
        # image_data = read_image(data_path)    
        t1ce  = np.array(read_image(data_path + '_T1C.nii.gz'), dtype='float32')  
        t1  = np.array(read_image(data_path + '_T1.nii.gz'), dtype='float32')
        t2  = np.array(read_image(data_path + '_T2.nii.gz'), dtype='float32')
        
        assert t1.shape == t2.shape , 't1, t2 shape not equal'
        assert np.abs(t1ce.shape[2] - t1.shape[2])<2 , 't1,  t1ce shape not equal'
        assert t1.shape[2] < 30, 't1 shape error'
        
        mask  = np.array(read_image(data_path + '_seg.nii.gz'), dtype='uint8') 
 
        roi_expand = roi_expand2(mask,expend_size_list=[25,25,1])
        w = max(roi_expand[0][1]-roi_expand[0][0],roi_expand[1][1]-roi_expand[1][0])
        h2 = math.ceil((w/w1)*16)
        h_r = roi_expand[2][1]-roi_expand[2][0]
        h_e = math.ceil((h2 - h_r)//2)
        if  t1.shape[2] > h2:
            if h2 > h_r:
                if roi_expand[2][1] + 1 + h_e > t1.shape[2]:
                    c_b = t1.shape[2] - h2
                    c_e = t1.shape[2]
                elif roi_expand[2][0] - h_e < 0:
                    c_b = 0 
                    c_e = h2
                else:
                    c_b = roi_expand[2][0] - h_e
                    c_e = roi_expand[2][1] + h_e  
            
            else:
                c_b =  roi_expand[2][0]             
                c_e =  roi_expand[2][1] + 1
        else:
            c_b = 0
            c_e = t1.shape[2]   

        if h2 > h_r:            
            assert  c_e-c_b - h2 <=1, 'c_e-c_b != h2' 

        t1_roi = t1[(roi_expand[0][1]+roi_expand[0][0])//2-w//2:(roi_expand[0][1]+roi_expand[0][0])//2+w//2,
                 (roi_expand[1][1]+roi_expand[1][0])//2-w//2:(roi_expand[1][1]+roi_expand[1][0])//2+w//2,
                  c_b:c_e]
        t1_roi = transform.resize(t1_roi,(w1, h, d), order=3) 

        t2_roi = t2[(roi_expand[0][1]+roi_expand[0][0])//2-w//2:(roi_expand[0][1]+roi_expand[0][0])//2+w//2,
                 (roi_expand[1][1]+roi_expand[1][0])//2-w//2:(roi_expand[1][1]+roi_expand[1][0])//2+w//2,
                  c_b:c_e]
        t2_roi = transform.resize(t2_roi,(w1, h, d), order=3) 

        t1ce_roi = t1ce[(roi_expand[0][1]+roi_expand[0][0])//2-w//2:(roi_expand[0][1]+roi_expand[0][0])//2+w//2,
                 (roi_expand[1][1]+roi_expand[1][0])//2-w//2:(roi_expand[1][1]+roi_expand[1][0])//2+w//2,
                  c_b:c_e]
        t1ce_roi = transform.resize(t1ce_roi,(w1, h, d), order=3) 


        item = self.transform({'t1':t1_roi,'t2':t2_roi,'t1ce':t1ce_roi})#, 'label':mask}) #,'t1ce':t1ce}

        # if self.mode == 'train':     # train
        #     item = item[0]   # [0] for RandCropByPosNegLabeld
        image = torch.permute(item['image'], (0, 3, 1, 2)) 
  
        # image = image.float()
       
       

        return (image, np.int8(data_label), data_name,idx)


    


