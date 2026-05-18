import os
import json

path = r'C:\Users\lahir\Downloads\ssv2_analysis\vjepa2-vitl-fpc16-256-ssv2_eval.txt'


with open(path, 'r') as f:
    lines = f.readlines()

correct = 0
total = 0
for line in lines:
    if line.split(':')[-1].replace('\n','').replace(' ','') == "True":
        correct+=1
    else:
        pass
    total += 1
print(f'accuracy : {correct/total*100}')
    