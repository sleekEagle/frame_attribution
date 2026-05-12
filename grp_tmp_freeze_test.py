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
from pathlib import Path
import torch.nn.functional as F
from scipy.stats import entropy

random.seed(78)
    
UCF_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'
OUT_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\grp_tmp_freeze'
thr = Path(UCF_PATH).stem.split('_')[-1]
OUT_PATH = os.path.join(OUT_PATH, f'{thr}.jsonl')

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
    # group_dict = {6: [0,1,2,3,4,5,6,7,8,9,10]}
    # n=6
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
    # group_dict = {2: [0, 1, 3, 4, 5], 6: [7], 8: [ 9, 10, 11, 12, 14,13], 15: []}
    # n=15
    group = func.deep_copy_dict(group_dict)

    frames = get_frames(group, n)
    idx = random.choice([0,-1])

    def mod_group(idx):
        global new_k
        new_k = -1
        add = -1 if idx==0 else 1
        search_frame = frames[idx]+add
        if search_frame<0 or search_frame>max_frame(group):
            return -1
        belong_k = which_group(search_frame, group)
        if n==frames[idx]:
            if len(group[n])>0: 
                new_k = random.choice(group[n])
                group[new_k] = group[n]
                group[new_k].remove(new_k)
            del group[n]
        else:
            group[n].remove(frames[idx])
            
        group[belong_k].append(frames[idx])
        group[belong_k].sort()

    ret = mod_group(idx)
    if ret==-1:
        idx = [val for val in [0,-1] if val!=idx][0]
        ret = mod_group(idx)
    
    return group, new_k

def grp_freeze():
    ucf101dm = func.UCF101_data_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ucf101dm.model
    model = model.to(device)
    model.eval()

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
            
            # increase sizes of all groups one by one
            N_ITR = 4

            if len(groups) == 1: continue 

            grp_dict = {}
            grp_dict['filename'] = filename
            grp_dict['original'] = groups

            d = {}
            for n_grp in groups:
                groups_ = func.deep_copy_dict(groups)
                grp_list = []
                for i in range(N_ITR):
                    groups_ = increase_size(groups_, n_grp)
                    if len(groups_) == 1:
                        break
                    grp_list.append(groups_)
                d[n_grp] = grp_list
            grp_dict['increasing'] = d

            # decrease sizes of all groups one by one
            d = {}
            for n_grp in groups:
                groups_ = func.deep_copy_dict(groups)
                grp_list = []
                for i in range(N_ITR):
                    if i == 0:
                        d_key = n_grp
                    groups_, new_k = decrease_size(groups_, n_grp)
                    if new_k!=-1:
                        n_grp = new_k
                    grp_list.append(groups_)
                    if n_grp not in groups_:
                        break
                d[d_key] = grp_list
            grp_dict['decreasing'] = d


            # create grouped videos
            vid_t = torch.empty(0)

            v_ = func.create_grouped_video(video.permute(1,0,2,3), grp_dict['original'])
            vid_t = torch.concat([vid_t,v_[None,:]], dim=0)

            for grp_id in grp_dict['increasing']:
                for g in grp_dict['increasing'][grp_id]:
                    v_ = func.create_grouped_video(video.permute(1,0,2,3), g)
                    vid_t = torch.concat([vid_t,v_[None,:]], dim=0)

            for grp_id in grp_dict['decreasing']:
                for g in grp_dict['decreasing'][grp_id]:
                    v_ = func.create_grouped_video(video.permute(1,0,2,3), g)
                    vid_t = torch.concat([vid_t,v_[None,:]], dim=0)

            vid_t = vid_t.to(device)
            with torch.no_grad():
                pred = model(vid_t)
            grp_dict['pred'] = pred.tolist()


            # analyze the pred changes
            pred = pred.cpu()
            pred_sm = F.softmax(pred, dim=1)
            entr = [float(entropy(p)) for p in pred_sm]
            entr_increase = [entr[0]-entr[i] for i in range(1, len(entr))]
            grp_dict['entropy_increase'] = entr_increase




            # save dict into a file
            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(grp_dict) + '\n')
                    

if __name__ == "__main__":
    grp_freeze()