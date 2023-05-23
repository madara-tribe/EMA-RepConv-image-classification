import torch
import torch.nn as nn
import torch.nn.functional as F

T = 0.01

def distilladtion_loss(llm_logits, logits, target):
    criterion = nn.CrossEntropyLoss(label_smoothing=0)
    mse = nn.MSELoss()    
    
    #llm_logits = llm_logits.detach()
    #pseudo_label = torch.argmax(F.softmax(llm_logits, dim=1), dim=1).long()
    #dist_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(F.log_softmax(logits, dim=1)/T, F.log_softmax(llm_logits, dim=1)/T) 
    #+ F.binary_cross_entropy(logits.sigmoid(), llm_logits.sigmoid())
    #dist_loss = F.binary_cross_entropy(logits.sigmoid(), llm_logits.sigmoid())
    #dist_loss = mse(logits, llm_logits)
    #print(pseudo_label.shape, target.shape)
    main_loss = criterion(logits, target) #pseudo_label)
    return main_loss

