import numpy as np





def  roi_expand( mask,expend_size_list,expend_size_prob):
    
    '''
    expend_size_list= [[5,100,100],[3,50,50],[1,15,15]]
    expend_size_prob= [0.2,0.5,0.7]
    '''
    mask1 = mask[0].numpy()
    expand_mask=np.zeros_like(mask1)
    expand_mask=np.float32(expand_mask)
    for idx in range(len(expend_size_list)):
        
        roi_expand_max = np.array(list(zip(*np.where(mask1>0)))) + np.array(expend_size_list[idx])
        roi_expand_min = np.array(list(zip(*np.where(mask1>0)))) - np.array([5,100,100])
        roi_expand = [(max(min(roi_expand_min[:,0]),0), min(max(roi_expand_max[:,0]),mask1.shape[0])),
                    (max(min(roi_expand_min[:,1]),0), min(max(roi_expand_max[:,1]),mask1.shape[1])),
                    (max(min(roi_expand_min[:,2]),0), min(max(roi_expand_max[:,2]),mask1.shape[2]))]   

        expand_mask[roi_expand[0][0]:roi_expand[0][1], roi_expand[1][0]:roi_expand[1][1], roi_expand[2][0]:roi_expand[2][1]] = expend_size_prob[idx]

        
    roi_expand_max = np.array(list(zip(*np.where(mask1>0))))
    roi_expand_min = np.array(list(zip(*np.where(mask1>0))))
    roi_expand = [(max(min(roi_expand_min[:,0]),0), min(max(roi_expand_max[:,0]),mask1.shape[0])),
                (max(min(roi_expand_min[:,1]),0), min(max(roi_expand_max[:,1]),mask1.shape[1])),
                (max(min(roi_expand_min[:,2]),0), min(max(roi_expand_max[:,2]),mask1.shape[2]))]
        
    expand_mask[np.where(mask1>0)] = 1
    expand_mask[expand_mask == 0] = 0.1    
    
    return expand_mask



import numpy as np





def  roi_expand2(mask,expend_size_list=[25,25,1]):
    
    '''
    expend_size_list= [[5,100,100],[3,50,50],[1,15,15]]
    expend_size_prob= [0.2,0.5,0.7]    '''
   
  
    roi_expand_max = np.array(list(zip(*np.where(mask>0)))) + np.array(expend_size_list)
    roi_expand_min = np.array(list(zip(*np.where(mask>0)))) - np.array(expend_size_list)   
    roi_expand = [(max(min(roi_expand_min[:,0]),0), min(max(roi_expand_max[:,0]),mask.shape[0])),
                  (max(min(roi_expand_min[:,1]),0), min(max(roi_expand_max[:,1]),mask.shape[1])),
                  (max(min(roi_expand_min[:,2]),0), min(max(roi_expand_max[:,2]),mask.shape[2]))] 
  
    
    return roi_expand

# class roi_expand():
#     def __init__(self, mask,expend_size_list,expend_size_prob):
#         mask1 = mask[0].numpy()
#         expand_mask=np.zeros_like(mask1)
#         expand_mask=np.float32(expand_mask)
#         for idx in range(len(expend_size_list)):
#             roi_expand_max = np.array(list(zip(*np.where(mask1>0)))) + np.array([5,100,100])
#             roi_expand_min = np.array(list(zip(*np.where(mask1>0)))) - np.array([5,100,100])
#             roi_expand = [(max(min(roi_expand_min[:,0]),0), min(max(roi_expand_max[:,0]),mask1.shape[0])),
#                         (max(min(roi_expand_min[:,1]),0), min(max(roi_expand_max[:,1]),mask1.shape[1])),
#                         (max(min(roi_expand_min[:,2]),0), min(max(roi_expand_max[:,2]),mask1.shape[2]))]   

#             expand_mask[roi_expand[0][0]:roi_expand[0][1], roi_expand[1][0]:roi_expand[1][1], roi_expand[2][0]:roi_expand[2][1]] = 0.2

#             roi_expand_max = np.array(list(zip(*np.where(mask1>0)))) + np.array([3,50,50])
#             roi_expand_min = np.array(list(zip(*np.where(mask1>0)))) - np.array([3,50,50])
#             roi_expand = [(max(min(roi_expand_min[:,0]),0), min(max(roi_expand_max[:,0]),mask1.shape[0])),
#                         (max(min(roi_expand_min[:,1]),0), min(max(roi_expand_max[:,1]),mask1.shape[1])),
#                         (max(min(roi_expand_min[:,2]),0), min(max(roi_expand_max[:,2]),mask1.shape[2]))]
                

#             expand_mask[roi_expand[0][0]:roi_expand[0][1], roi_expand[1][0]:roi_expand[1][1], roi_expand[2][0]:roi_expand[2][1]] = 0.5

#             roi_expand_max = np.array(list(zip(*np.where(mask1>0)))) + np.array([1,15,15])
#             roi_expand_min = np.array(list(zip(*np.where(mask1>0)))) - np.array([1,15,15])
#             roi_expand = [(max(min(roi_expand_min[:,0]),0), min(max(roi_expand_max[:,0]),mask1.shape[0])),
#                         (max(min(roi_expand_min[:,1]),0), min(max(roi_expand_max[:,1]),mask1.shape[1])),
#                         (max(min(roi_expand_min[:,2]),0), min(max(roi_expand_max[:,2]),mask1.shape[2]))]
                

#             expand_mask[roi_expand[0][0]:roi_expand[0][1], roi_expand[1][0]:roi_expand[1][1], roi_expand[2][0]:roi_expand[2][1]] = 0.7

#             roi_expand_max = np.array(list(zip(*np.where(mask1>0))))
#             roi_expand_min = np.array(list(zip(*np.where(mask1>0))))
#             roi_expand = [(max(min(roi_expand_min[:,0]),0), min(max(roi_expand_max[:,0]),mask1.shape[0])),
#                         (max(min(roi_expand_min[:,1]),0), min(max(roi_expand_max[:,1]),mask1.shape[1])),
#                         (max(min(roi_expand_min[:,2]),0), min(max(roi_expand_max[:,2]),mask1.shape[2]))]
                
#             expand_mask[np.where(mask1>0)] = 1
            
        
#         return expand_mask