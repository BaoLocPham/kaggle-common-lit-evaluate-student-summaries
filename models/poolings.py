import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e9
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
    
class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()
        
        self.mean_pooler = MeanPooling()
        self.max_pooler  = MaxPooling()
        
    def forward(self, last_hidden_state, attention_mask):
        mean_pooler = self.mean_pooler( last_hidden_state ,attention_mask )
        max_pooler =  self.max_pooler( last_hidden_state ,attention_mask )
        out = torch.concat([mean_pooler ,max_pooler ] , 1)
        return out

class GeMText(nn.Module):
    def __init__(self, dim = 1, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1

    def forward(self, last_hidden_state, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        x = (last_hidden_state.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret
    
def get_pooling_layer(cfg):
    if cfg.pooling == 'Mean':
        return MeanPooling()
    
    elif cfg.pooling == 'Max':
        return MaxPooling()
    
    elif cfg.pooling == 'MeanMax':
        return MeanMaxPooling()
    
    elif cfg.pooling == 'GemText':
        return GeMText()