import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import AutoModel, AutoTokenizer, AdamW, DataCollatorWithPadding, AutoConfig, AutoTokenizer, logging
from .poolings import get_pooling_layer
from .freezing import top_half_layer_freeze

class CommontLitModel(nn.Module):
    def __init__(self, model_name, cfg ):
        super(CommontLitModel, self).__init__()
        
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.config = AutoConfig.from_pretrained(cfg.model_name)
        #self.drop = nn.Dropout(p=0.2)
        self.pooler = get_pooling_layer(cfg=cfg)

        if cfg.pooling == 'MeanMax':
            self.fc = nn.Linear(2*self.config.hidden_size, 2)
        else:
            self.fc = nn.Linear(self.config.hidden_size, 2)
            
        
        self._init_weights(self.fc)
        
        if cfg.freezing:
            top_half_layer_freeze(self.model)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
           
    def forward(self, ids, mask):
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        #out = self.drop(out)
        outputs = self.fc(out)
        return outputs