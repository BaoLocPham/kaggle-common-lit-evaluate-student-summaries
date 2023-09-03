
def freeze(module):
    """
    Freezes module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = False

def odd_layer_freeze(module):
    for i in range(1,24,2):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False
            
def even_layer_freeze(module):
    for i in range(0,24,2):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False
            
def top_half_layer_freeze(module):
    for i in range(0,13,1):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False

def bottom_half_layer_freeze(module):
    for i in range(13,14,1):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False