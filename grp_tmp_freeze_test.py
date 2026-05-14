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
import matplotlib.pyplot as plt

random.seed(78)

KMAX = 3 # number of different margins
N_ITR = 4
UCF_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'
OUT_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\grp_tmp_freeze'
thr = Path(UCF_PATH).stem.split('_')[-1]
OUT_PATH = os.path.join(OUT_PATH, f'{thr}.jsonl')
IMP_PATH = os.path.join(Path(UCF_PATH).parent , 'shap', f'exactSHAP_future_{thr}.jsonl')

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
    # group_dict = {1:[0,2,3], 6: [4,5,7,8], 9:[], 11:[10], 12:[13,14,15]}
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
        if len(group[belong_k]) == 0: 
            return -1
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
                    if groups_ == -1:
                        break
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

            #margin difference between original pred and the grouped
            margin_dict = {}
            #margins = 1,2,3 : 1: highest - 2nd highest  2: highest - mean(2nd , 3rd)
            for k in range(1,KMAX+1):
                margins = [float(func.get_margin(p, k=k)) for p in pred]
                marg_diffs = [margin - margins[0] for margin in margins[1:]]
                margin_dict[k] = marg_diffs
            grp_dict['margin_diff'] = margin_dict

            #entropy difference
            pred_sm = F.softmax(pred, dim=1)
            entr = [float(entropy(p)) for p in pred_sm]
            entr_increase = [entr[0]-entr[i] for i in range(1, len(entr))]
            grp_dict['entropy_increase'] = entr_increase

            # save dict into a file
            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(grp_dict) + '\n')
                    

def create_plot():
    ucf101dm = func.UCF101_data_model()
    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k

    with open(IMP_PATH, 'r') as f:
        data = [json.loads(line) for line in f]
    
    sv_dict = {}
    for d in data:
        f = d['filename']
        cls_idx = class_labels[f.split('_')[1].lower()]
        g = [int(k) for k in list(d['groups'].keys())]
        sv = d['shapley_values'][0]

        sv_g = []
        for i in range(len(g)):
            sv_g.append(sv[i][cls_idx])

        sv_dict[f] = {'groups': g, 'shapley_values': sv_g}
    del data

    #init metric dictionary
    met = {
        'margin_diff': {k: [] for k in range(1,KMAX+1)},
        'entropy_increase': []
    }
    all_met_increasing = {}
    for i in range(11):
        d_ = {}
        for it in range(N_ITR):
            d_[it] = func.deep_copy_dict(met)
        all_met_increasing[i/10] = d_    

    all_met_decreasing = {}
    for i in range(11):    
        d_ = {}
        for it in range(N_ITR):
            d_[it] = func.deep_copy_dict(met)
        all_met_decreasing[i/10] = d_    


    n=0
    with open(OUT_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))
    with open(OUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.1f}% is done.', end='\r')
            n+=1
            line = line.strip()
            record = json.loads(line)
            sv = sv_dict[record['filename']]

            #sanity check
            assert sorted(sv['groups']) == sorted( [int(k) for k in list(record['original'].keys())]), 'Groups do not match'

            sort_idx = np.argsort(np.array(sv['shapley_values']))
            asc_grps = np.array(sv['groups'])[sort_idx] # groups ordered in ascending order of importance
            asc_grps = [int(g) for g in asc_grps]

            sv_norm = sv['shapley_values']
            sv_norm = (sv_norm - np.min(sv_norm)) / (np.max(sv_norm) - np.min(sv_norm))
            asc_sv_norm = np.array(sv_norm)[sort_idx]

            i = 0
            metrics = {}

            met = {}
            for g in record['increasing'].keys():
                met_ = {}
                for idx in range(len(record['increasing'][g])):
                    m_ = {}
                    margin_ks = record['margin_diff'].keys()
                    for mk in margin_ks:
                        m_[f'margin_diff_{mk}'] = record['margin_diff'][mk][i]
                    m_[f'entropy_increase'] = record['entropy_increase'][i]
                    met_[idx] = m_
                    i+=1
                met[g] = met_
            metrics['increasing'] = met

            met = {}
            for g in record['decreasing'].keys():
                met_ = {}
                for idx in range(len(record['decreasing'][g])):
                    m_ = {}
                    margin_ks = record['margin_diff'].keys()
                    for mk in margin_ks:
                        m_[f'margin_diff_{mk}'] = record['margin_diff'][mk][i]
                    m_[f'entropy_increase'] = record['entropy_increase'][i]
                    met_[idx] = m_
                    i+=1
                met[g] = met_
            metrics['decreasing'] = met


            for th, g in enumerate(asc_grps):
                val = metrics['increasing'][str(g)]
                key = int(float(asc_sv_norm[th])*10)/10

                for step in val.keys():
                    all_met_increasing[key][step]['margin_diff'][1].append(val[step]['margin_diff_1'])
                    all_met_increasing[key][step]['margin_diff'][2].append(val[step]['margin_diff_2'])
                    all_met_increasing[key][step]['margin_diff'][3].append(val[step]['margin_diff_3'])
                    all_met_increasing[key][step]['entropy_increase'].append(val[step]['entropy_increase'])
                
                val = metrics['decreasing'][str(g)]
                for step in val.keys():
                    all_met_decreasing[key][step]['margin_diff'][1].append(val[step]['margin_diff_1'])
                    all_met_decreasing[key][step]['margin_diff'][2].append(val[step]['margin_diff_2'])
                    all_met_decreasing[key][step]['margin_diff'][3].append(val[step]['margin_diff_3'])
                    all_met_decreasing[key][step]['entropy_increase'].append(val[step]['entropy_increase'])



    #calculate metric means, stds
    stat = {
        'margin_diff': {k: {'mean': -1, 'std': -1, 'count': 0} for k in range(1,KMAX+1)},
        'entropy_increase': {'mean': -1, 'std': -1, 'count': 0}
    }
    stat_increasing = {}
    for i in range(11):
        d_ = {}
        for it in range(N_ITR):
            d_[it] = func.deep_copy_dict(stat)
        stat_increasing[i/10] = d_    

    stat_decreasing = {}
    for i in range(11):    
        d_ = {}
        for it in range(N_ITR):
            d_[it] = func.deep_copy_dict(stat)
        stat_decreasing[i/10] = d_ 


    for k in all_met_increasing:
        for step in all_met_increasing[k]:
            #increasing
            stat_increasing[k][step]['entropy_increase']['mean'] = np.mean(all_met_increasing[k][step]['entropy_increase'])
            stat_increasing[k][step]['entropy_increase']['std'] = np.std(all_met_increasing[k][step]['entropy_increase'])
            stat_increasing[k][step]['entropy_increase']['count']+=len(all_met_increasing[k][step]['entropy_increase'])
            for mk in range(1,KMAX+1):
                stat_increasing[k][step][f'margin_diff'][mk]['mean'] = np.mean(all_met_increasing[k][step]['margin_diff'][mk])
                stat_increasing[k][step][f'margin_diff'][mk]['std'] = np.std(all_met_increasing[k][step]['margin_diff'][mk])
                stat_increasing[k][step][f'margin_diff'][mk]['count'] += len(all_met_increasing[k][step]['margin_diff'][mk])
            #decreasing
            stat_decreasing[k][step]['entropy_increase']['mean'] = np.mean(all_met_decreasing[k][step]['entropy_increase'])
            stat_decreasing[k][step]['entropy_increase']['std'] = np.std(all_met_decreasing[k][step]['entropy_increase'])
            stat_decreasing[k][step]['entropy_increase']['count'] += len(all_met_decreasing[k][step]['entropy_increase'])
            for mk in range(1,KMAX+1):  
                stat_decreasing[k][step][f'margin_diff'][mk]['mean'] = np.mean(all_met_decreasing[k][step]['margin_diff'][mk])
                stat_decreasing[k][step][f'margin_diff'][mk]['std'] = np.std(all_met_decreasing[k][step]['margin_diff'][mk])
                stat_decreasing[k][step][f'margin_diff'][mk]['count'] += len(all_met_decreasing[k][step]['margin_diff'][mk])

    #plot metrics
    imps = list(all_met_increasing.keys())

    def im_metrics(im):
        steps = list(range(1,5))
        steps = [-1*i for i in steps[::-1]] + steps
        marg1_m, marg2_m, marg3_m = [], [], []
        entr_m = []
        marg1_std, marg2_std, marg3_std = [], [], []
        entr_std = []
        for step in steps:
            if step < 0:
                idx = step*-1 -1
                val = stat_decreasing[im][idx]
                marg1_m.append(val['margin_diff'][1]['mean'])
                marg2_m.append(val['margin_diff'][2]['mean'])
                marg3_m.append(val['margin_diff'][3]['mean'])
                entr_m.append(val['entropy_increase']['mean'])
                marg1_std.append(val['margin_diff'][1]['std'])
                marg2_std.append(val['margin_diff'][2]['std'])  
                marg3_std.append(val['margin_diff'][3]['std'])
                entr_std.append(val['entropy_increase']['std'])
            else:
                idx = step -1
                val = stat_increasing[im][idx]
                marg1_m.append(val['margin_diff'][1]['mean'])
                marg2_m.append(val['margin_diff'][2]['mean'])
                marg3_m.append(val['margin_diff'][3]['mean'])
                entr_m.append(val['entropy_increase']['mean'])
                marg1_std.append(val['margin_diff'][1]['std'])
                marg2_std.append(val['margin_diff'][2]['std'])  
                marg3_std.append(val['margin_diff'][3]['std'])
                entr_std.append(val['entropy_increase']['std'])
        metrics = {
            'steps': steps,
            'marg1_m': marg1_m,
            'marg2_m': marg2_m,
            'marg3_m': marg3_m,
            'entr_m': entr_m,
            'marg1_std': marg1_std,
            'marg2_std': marg2_std,
            'marg3_std': marg3_std,
            'entr_std': entr_std,
        }
        return metrics
    
    PLOT_DIR = r'C:\Users\lahir\Downloads\UCF101\analysis\plots\grouping\grp_size_change\margin3'

    plt.ioff() 
    for imp in imps:
        metrics = im_metrics(imp)
        plt.figure(figsize=(10, 6))
        plt.errorbar(metrics['steps'], metrics['marg3_m'], yerr=metrics['marg3_std'], 
                    fmt='o-',           # line with circles
                    capsize=5,          # size of error bar caps
                    capthick=2,         # thickness of caps
                    elinewidth=2,       # thickness of error bars
                    color='green',      # color of everything
                    ecolor='red',       # color of error bars only
                    label='Mean ± Std')
        plt.xlabel('Step')
        plt.ylabel('Margin 3 Change')
        plt.title(f'Importance={imp}')
        # plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOT_DIR, f'imp_{imp}.png'), dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    #all on the same plot
    PLOT_DIR = r'C:\Users\lahir\Downloads\UCF101\analysis\plots\grouping\grp_size_change\entropy_change'

    colormap = plt.cm.viridis
    norm = plt.Normalize(min(imps), max(imps))
    legend_handles = []
    for imp in imps:
        metrics = im_metrics(imp)
        color = colormap(norm(imp))
        line, = plt.plot(metrics['steps'], metrics['entr_m'], marker='o', color=color, label=f'{imp}', markersize=3, linewidth=1)
        legend_handles.append(line)
    
    # Add colorbar
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Steps')
    plt.ylabel('Entropy Increase')
    plt.title('Entropy Increase Evolution by Importance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'evolution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    create_plot()