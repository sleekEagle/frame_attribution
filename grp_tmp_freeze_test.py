import shap
import numpy as np
import func
import json
import torch    
import os
from pathlib import Path
import CONST
import func
import random
random.seed(78)
    
UCF_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'


def get_frames(group, n):
    assert n in group, 'n must be a key in the group dict'
    ret = group[n] + [n]
    ret.sort()
    return ret

def max_frame(group_dict):
    m = 0
    for k in group_dict:
        m_ = max(get_frames(group_dict, k))
        if m_>m:
            m = m_
    return m

def which_group(f, group):
    for k in group:
        if f in group[k] or f==k:
            return k

def increase_size(group_dict, n):
    # group_dict = {2: [0, 1, 3, 4, 5], 6: [], 13: [7, 8, 9, 10, 11, 12, 14], 15: []}
    # n=13
    group = func.deep_copy_dict(group_dict)

    frames = get_frames(group, n)
    idx = random.choice([0,-1])

    def mod_group(idx):
        add = -1 if idx==0 else 1
        search_frame = frames[idx]+add
        if search_frame<0 or search_frame>max_frame(group):
            return -1
        belong_k = which_group(search_frame, group)
        if belong_k==search_frame: # key is the frame we are looking for
            if len(group[search_frame])>0: 
                new_k = random.choice(group[search_frame])
                group[new_k] = group[search_frame]
                group[new_k].remove(new_k)
            del group[search_frame]
        else:
            group[belong_k].remove(search_frame)
        group[n].append(search_frame)
        group[n].sort()

    ret = mod_group(idx)
    if ret==-1:
        idx = [val for val in [0,-1] if val!=idx][0]
        ret = mod_group(idx)
    
    return group

def decrease_size(group_dict, n):
    # group_dict = {2: [0, 1, 3, 4, 5], 6: [], 13: [7, 8, 9, 10, 11, 12, 14], 15: []}
    # n=13
    group = func.deep_copy_dict(group_dict)

    frames = get_frames(group, n)
    idx = random.choice([0,-1])

    def mod_group(idx):
        add = -1 if idx==0 else 1
        search_frame = frames[idx]+add
        if search_frame<0 or search_frame>max_frame(group):
            return -1
        belong_k = which_group(search_frame, group)
        if belong_k==search_frame: # key is the frame we are looking for
            if len(group[search_frame])>0: 
                new_k = random.choice(group[search_frame])
                group[new_k] = group[search_frame]
                group[new_k].remove(new_k)
            del group[search_frame]
        else:
            group[belong_k].remove(search_frame)
        group[n].append(search_frame)
        group[n].sort()

    ret = mod_group(idx)
    if ret==-1:
        idx = [val for val in [0,-1] if val!=idx][0]
        ret = mod_group(idx)
    
    return group

def grp_freeze():
    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model

    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k

    #read groups
    n=0
    with open(UCF_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))
    with open(UCF_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.1f}% is done.', end='\r')
            n+=1
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            filename = record['filename']
            # if filename!='v_ApplyEyeMakeup_g01_c01':
            #     continue
            p = ucf101dm.construct_vid_path_from_full(filename)
            video = ucf101dm.load_jpg_ucf101(p, n=0)
            g = record['groups']
            groups = {}
            for k in g:
                if 'frames' in g[k]:
                    f = g[k]['frames']
                else: 
                    f = []
                groups[int(k)] = f

            groups = increase_size(groups, 1)
            
            for n in groups:
                groups_ = increase_size(groups, n)
            pass



if __name__ == "__main__":
    grp_freeze()