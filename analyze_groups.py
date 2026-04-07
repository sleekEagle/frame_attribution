import os
from glob import glob
import func
import json
import numpy as np


def UCF101_minchange():
    path = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.01.jsonl'
    in_grp = []
    out_grp = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            d = record['groups']
            for k in d:
                if len(d[k])==0: 
                    continue
                l = d[k]['min_change_list']
                if len(d[k]['frames']) == 0: 
                    continue
                if len(l)==1 and int(k) in [14,15]:
                    continue
                assert not (len(l)==1 and int(k) not in [14,15]), 'error'

                in_grp_changes = l[:-1]
                out_grp_changes = l[-1]
                in_grp.append(in_grp_changes)
                out_grp.append(out_grp_changes)

    in_d_l = []
    out_d_l = []
    for i in range(len(in_grp)):
        if len(in_grp[i])<=1: continue
        in_diffs = [in_grp[i][j+1]-in_grp[i][j] for j in range(len(in_grp[i])-1)]
        in_diff = sum(in_diffs)/len(in_diffs)
        out_diff = out_grp[i] - in_grp[i][-1]
        in_d_l.append(in_diff)
        out_d_l.append(out_diff)
    
    print(f'mean in_diff: {np.mean(in_d_l)}, mean out_diff: {np.mean(out_d_l)}')
    print(f'std in_diff: {np.std(in_d_l)}, std out_diff: {np.std(out_d_l)}')

if __name__ == '__main__':
    UCF101_minchange()