import os
import json
import torch
import torch.nn.functional as F
from scipy.stats import entropy



# path = r'C:\Users\lahir\Downloads\ssv2_analysis\vjepa2-vitl-fpc16-256-ssv2_eval.txt'


# with open(path, 'r') as f:
#     lines = f.readlines()

# correct = 0
# total = 0
# for line in lines:
#     if line.split(':')[-1].replace('\n','').replace(' ','') == "True":
#         correct+=1
#     else:
#         pass
#     total += 1
# print(f'accuracy : {correct/total*100}')
    

# pred_p = torch.rand(100)
# pred_p = F.softmax(pred_p,dim=0)
# o_entr = -torch.sum(pred_p * torch.log(pred_p + 1e-10), dim=-1) 
# entropy(pred_p)
