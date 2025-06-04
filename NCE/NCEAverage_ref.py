import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import numpy as np
import torch.nn.functional as F


class NCEAverage_each_pre(nn.Module):
    # NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    def __init__(self, inputSize, gpu_ids, g_data, m_data, memory_feat_gbm, memory_prob_gbm, memory_name_gbm, memory_feat_mate, memory_prob_mate, memory_name_mate, 
                 K, T=0.07, momentum=0.5, use_softmax=False):
        # args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax
        super(NCEAverage_each_pre, self).__init__()
        self.gpu_ids = gpu_ids
        self.g_data = g_data
        self.unigrams_p_g = torch.ones(self.g_data//4)
        # self.unigrams_p = torch.ones(self.p_data//4)
        self.multinomial_p_g = AliasMethod(self.unigrams_p_g)
        self.multinomial_p_g.cuda(self.gpu_ids)
        
        self.unigrams_n_g = torch.ones(self.g_data)
        self.multinomial_n_g = AliasMethod(self.unigrams_n_g )
        self.multinomial_n_g.cuda(self.gpu_ids)

                
        self.m_data =m_data
        self.unigrams_p_m = torch.ones(self.m_data//4)
        # self.unigrams_p = torch.ones(self.p_data//4)
        self.multinomial_p_m = AliasMethod(self.unigrams_p_m)
        self.multinomial_p_m.cuda(self.gpu_ids)
        
        self.unigrams_n_m = torch.ones(self.m_data)
        self.multinomial_n_m = AliasMethod(self.unigrams_n_m )
        self.multinomial_n_m.cuda(self.gpu_ids)
        
        
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        # stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_feat_gbm', memory_feat_gbm)
        self.register_buffer('memory_prob_gbm', memory_prob_gbm)
        self.memory_name_gbm = memory_name_gbm

        self.register_buffer('memory_feat_mate', memory_feat_mate)
        self.register_buffer('memory_prob_mate', memory_prob_mate)
        self.memory_name_mate= memory_name_mate

        self.memory_feat_gbm = nn.functional.normalize(self.memory_feat_gbm, dim=1)
        self.memory_feat_mate = nn.functional.normalize(self.memory_feat_mate, dim=1)
        
    def forward(self, feat, label, out, name, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        feat = nn.functional.normalize(feat, dim=1)

        momentum = self.params[4].item()
        batchSize = feat.size(0)
        outputSize = self.memory_feat_gbm.size(0)
        inputSize = self.memory_feat_gbm.size(1)
        
        # score computation
        if idx is None:
            idx_n_g = self.multinomial_n_g.draw(self.K * (1)).view(self.K, -1)
            idx_p_g = self.multinomial_p_g.draw(1).view(1, -1)
            idx_n_m = self.multinomial_n_m.draw(self.K * (1)).view(self.K, -1)
            idx_p_m = self.multinomial_p_m.draw(1).view(1, -1)
            # idx.select(1, 0).copy_(y.data)
        
        index_gbm = torch.nonzero((label == 0).int(), as_tuple=True)[0]
        # sample
        out_feature_p_n = torch.zeros(batchSize, K +1 ).cuda(self.gpu_ids)
        for ind in index_gbm:
            feat_c = feat[ind].view(1, inputSize)

            weight_feature_n = torch.index_select(self.memory_feat_mate, 0, idx_n_m.view(-1)).detach()
            weight_feature_p = torch.index_select(self.memory_feat_gbm, 0, idx_p_g.view(-1)).detach()
        
            # l_pos = torch.einsum("nc,nc->n", [feat_c,weight_feature_p]).unsqueeze(-1)
            l_pos = F.cosine_similarity(feat_c, weight_feature_p, dim=1)
            # l_neg = torch.einsum("nc,ck->nk", [feat_c, weight_feature_n.view(inputSize, -1)])
            l_neg = F.cosine_similarity(feat_c, weight_feature_n, dim=1)
        
            out_feature_p_n [ind,:]  = torch.cat([l_pos, l_neg], dim=0)
       
        # sample
        index_mate = torch.nonzero((label == 1).int(), as_tuple=True)[0]
        for ind in index_mate:
            feat_c = feat[ind].view(1, inputSize)
            weight_feature_n = torch.index_select(self.memory_feat_gbm, 0, idx_n_g.view(-1)).detach()
            weight_feature_p = torch.index_select(self.memory_feat_mate, 0, idx_p_m.view(-1)).detach()
            # # weight_feature_n = torch.index_select(self.memory_feat_mate, 0, idx_n.view(-1)).detach()
            # # weight_feature_p = torch.index_select(self.memory_feat_gbm, 0, idx_p.view(-1)).detach()
        
            # l_pos = torch.einsum("nc,nc->n", [feat_c,weight_feature_p]).unsqueeze(-1)
            l_pos = F.cosine_similarity(feat_c, weight_feature_p, dim=1)
            # l_neg = torch.einsum("nc,ck->nk", [feat_c, weight_feature_n.view(inputSize, -1)])
            l_neg = F.cosine_similarity(feat_c, weight_feature_n, dim=1)
        
            out_feature_p_n [ind,:]  = torch.cat([l_pos, l_neg], dim=0)
        
        if self.use_softmax:
            out_feature_p_n = torch.div(out_feature_p_n, T)            
            out_feature_p_n = out_feature_p_n.contiguous()
        
        else:
            out_feature_p_n = torch.exp(torch.div(out_feature_p_n, T))
      
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_feature_p_n.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))

            out_feature_p_n = torch.div(out_feature_p_n, Z_l).contiguous()
     
        index_clu_gbm  = torch.nonzero((self.memory_prob_gbm > 0.5).int(), as_tuple=True)[0] 
        index_clu_mate = torch.nonzero((self.memory_prob_mate > 0.6).int(), as_tuple=True)[0]
        feature_clu_gbm =  torch.index_select(self.memory_feat_gbm,  0, index_clu_gbm.view(-1)).detach()
        feature_clu_mate = torch.index_select(self.memory_feat_mate, 0, index_clu_mate.view(-1)).detach()
        feature_clu_all = torch.cat([feature_clu_gbm, feature_clu_mate,feat], dim=0)
        gt_clu_all = torch.cat([torch.zeros_like(index_clu_gbm), torch.ones_like(index_clu_mate), label], dim=0)
     
        # # update memory
        # with torch.no_grad():
        #     for id in range(len(name)):
        #         if label[id] == 0:
        #             index = self.memory_name_gbm.index(name[id])               
        #             pos_prob = torch.index_select(self.memory_prob_gbm, 0, torch.tensor(index).cuda(self.gpu_ids))

        #             if out[id][label[id]] > 0.5 :
        #                 memory_feat_up = torch.mul(self.memory_feat_gbm[index],  momentum) + torch.mul(feat[id], 1 - momentum)
        #                 # memory_gbm_up_norm = memory_feat_up.pow(2).sum(0, keepdim=True).pow(0.5)
        #                 # updated_gbm = memory_feat_up.div(memory_gbm_up_norm)
        #                 self.memory_feat_gbm[index]=memory_feat_up.clone().detach()
        #                 self.memory_prob_gbm[index]= out[id][label[id]].clone().detach()
        #         else:
        #             index = self.memory_name_mate.index(name[id])      
        #             pos_prob = torch.index_select(self.memory_prob_mate, 0, torch.tensor(index).cuda(self.gpu_ids))

        #             if out[id][label[id]] > 0.5 :
        #                 memory_feat_up = torch.mul(self.memory_feat_mate[index],  momentum) + torch.mul(feat[id], 1 - momentum)
        #                 # memory_gbm_up_norm = memory_gbm_up.pow(2).sum(0, keepdim=True).pow(0.5)
        #                 # updated_gbm = memory_gbm_up.div(memory_gbm_up_norm)
        #                 self.memory_feat_mate[index]=memory_feat_up.clone().detach()
        #                 self.memory_prob_mate[index]= out[id][label[id]].clone().detach()
                
            
        #     sorted_id = sorted(range(len(self.memory_prob_gbm)), key=lambda k: self.memory_prob_gbm[k], reverse=True)
        #     self.memory_feat_gbm = torch.tensor(np.array([ list(self.memory_feat_gbm)[i].cpu() for i in sorted_id])).cuda(self.gpu_ids)
        #     self.memory_prob_gbm = torch.tensor(np.array([ list(self.memory_prob_gbm)[i].cpu() for i in sorted_id])).cuda(self.gpu_ids)
        #     self.memory_name_gbm = [ list(self.memory_name_gbm)[i] for i in sorted_id]            

        #     sorted_id = sorted(range(len(self.memory_prob_mate)), key=lambda k: self.memory_prob_mate[k], reverse=True)
        #     self.memory_feat_mate = torch.tensor(np.array([ list(self.memory_feat_mate)[i].cpu() for i in sorted_id])).cuda(self.gpu_ids)
        #     self.memory_prob_mate = torch.tensor(np.array([ list(self.memory_prob_mate)[i].cpu() for i in sorted_id])).cuda(self.gpu_ids)
        #     self.memory_name_mate = [ list(self.memory_name_mate)[i] for i in sorted_id]  

        return out_feature_p_n, feature_clu_all, gt_clu_all


# =========================
# InsDis and MoCo
# =========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda(self.gpu_ids)
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda(self.gpu_ids)
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out

# if __name__ == '__main__':
#     NCE=NCEAverage_PRE(128, 400,10,0.07, 0.5, True)
#     X=torch.randn(4,128)
#     Y=torch.randn(4,128)
#     a=NCE(X,Y,None)