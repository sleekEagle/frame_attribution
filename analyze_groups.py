import os
from glob import glob
from xml.sax.handler import all_features

from matplotlib import path
import func
import json
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
import random
from scipy.stats import entropy
from models.ssv2 import VJEPA2
from dataloaders import ssv2
import CONST
from pathlib import Path

UCF_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'

'''
*************************************************************************************
histogram-based similarity calculation for frames from the video
calculates the inter-group similarity and intra-group similarities
*************************************************************************************
'''
def frame_similarity_hist(frame1, frame2):

    frame1 = (frame1 - frame1.min())/(frame1.max()-frame1.min()+1e-5)
    frame2 = (frame2 - frame2.min())/(frame2.max()-frame2.min()+1e-5)

    bins = 32
    def compute_histogram(frame, bins):
        frame_quantized = (frame * (bins - 1)).long()  # [C, H, W]
        # Create 3D histogram index: R * bins^2 + G * bins + B
        r, g, b = frame_quantized[0], frame_quantized[1], frame_quantized[2]
        indices = r * (bins * bins) + g * bins + b
        hist = torch.bincount(indices.flatten(), minlength=bins**3)
        hist.float()
        return hist
    hist1 = compute_histogram(frame1, bins)
    hist2 = compute_histogram(frame2, bins)
    # Normalize histograms
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # Compute correlation (equivalent to cv2.HISTCMP_CORREL)
    mean1, mean2 = hist1.mean(), hist2.mean()
    numerator = ((hist1 - mean1) * (hist2 - mean2)).sum()
    denominator = torch.sqrt(((hist1 - mean1)**2).sum() * ((hist2 - mean2)**2).sum())
    similarity = numerator / denominator

    return similarity.item()

def video_similarity_hist(video, groups, MAX_COMBS=5):
    rep_frames = [int(k) for k in groups.keys()]
    out_combs = list(combinations(rep_frames, 2))
    in_combs = []
    for k in groups:
        if 'frames' not in groups[k]:
            continue
        frames = groups[k]['frames']
        grp_combs = [(int(k), f) for f in frames]
        in_combs.extend(grp_combs)
    n_samples = min(MAX_COMBS, min(len(in_combs),len(out_combs)))
    if n_samples==0:
        return 0, 0
    in_samples = random.sample(in_combs, n_samples)
    out_samples = random.sample(out_combs, n_samples)

    def mean_sim(samples):
        sim = 0
        for s in samples:
            frame1 = video[s[0]]
            frame2 = video[s[1]]
            s = frame_similarity_hist(frame1, frame2)
            sim+=s
        mean_sim = sim/len(samples)
        return mean_sim
    in_sim = mean_sim(in_samples)
    out_sim = mean_sim(out_samples)

    return in_sim, out_sim


def UCF101_metrics():
    path = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.01.jsonl'
    threshold = os.path.basename(path).split('_')[1][:5]
    n_g = 0
    entr_change = 0
    logit_change = 0
    N_MARGINS = 6
    margin_dict = {k: 0 for k in range(1, N_MARGINS+1)}
    n = 0
    in_sim, out_sim = 0, 0
    with open(path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))    

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.0f}% is done', end='\r')
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record['correct']: continue

            n_g += len(record['groups'].keys())

            for k in range(1,N_MARGINS+1):
                margin_dict[k] += record['all_group_change'][f'margin_{k}_change']

            #calc entropy difference
            # o_prob = F.softmax(torch.tensor(record['original_stat']['logits']),dim=0)
            # o_entr = entropy(o_prob)
            # g_prob = F.softmax(torch.tensor(record['all_grp_stats']['logits']),dim=0)
            # g_entr = entropy(g_prob)
            # entr_change_ = (o_entr - g_entr)/o_entr
            entr_change += record['all_group_change']['entropy_change']
            logit_change += record['all_group_change']['max_logit_change']


            #inter and intra group similarities
            # filename = ucf101dm.construct_vid_path_from_full(record['filename'])
            # if not filename=='C:\\Users\\lahir\\Downloads\\UCF101\\jpgs\\Archery\\v_Archery_g06_c02':
            #     continue
            # video = ucf101dm.load_jpg_ucf101(filename, n=0)
            # groups = record['groups']
            # if len(groups.keys())<=1: continue
            # is_, os_ = video_similarity_hist(video, groups)
            # in_sim += is_
            # out_sim += os_

            n += 1
        
        margin_dict = {k: float(margin_dict[k]/n) for k in range(1, N_MARGINS+1)}
        entr_change /= n
        n_g /= n
        logit_change /= n
        # in_sim /= n
        print(f'Threshold = {threshold}')
        print(f'average number of groups: {n_g} \naverage margin change: {margin_dict} \naverage entropy change: {entr_change} \n Max logit change: {logit_change}')
        print(f'n = {n}, total = {line_count}')
        # print(f'inter-group similarity: {out_sim}')
        # print(f'intra-group similarity: {in_sim}')

def calc_metrics():
    GRP_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\groups\groups_0.01.jsonl'
    threshold = os.path.basename(GRP_PATH).split('_')[1][:-6]
    n_g = 0
    entr_change = 0
    logit_change = 0
    N_MARGINS = 6
    grp_pred_correct = 0
    margin_dict = {k: 0 for k in range(1, N_MARGINS+1)}
    n = 0
    in_sim, out_sim = 0, 0
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))    

    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.0f}% is done', end='\r')
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # if not record['correct']: continue

            grp_pred_cls = record['grp_pred_cls']
            pred_cls = record['original_stat']['cls']
            if grp_pred_cls==pred_cls:
                grp_pred_correct+=1
            else: continue

            n_g += len(record['groups'].keys())

            for k in range(1,N_MARGINS+1):
                margin_dict[k] += record['all_group_change'][f'margin_{k}_change']

            entr_change += record['all_group_change']['entropy_change']
            logit_change += record['all_group_change']['max_logit_change']

            n += 1
        
        margin_dict = {k: float(margin_dict[k]/n) for k in range(1, N_MARGINS+1)}
        entr_change /= n
        n_g /= n
        logit_change /= n

        print(f'Threshold = {threshold}')
        print(f'average number of groups: {n_g} \naverage margin change: {margin_dict} \naverage entropy change: {entr_change} \n Max logit change: {logit_change}')
        print(f'group prediction accuracy: {grp_pred_correct/n:.2%}')
        print(f'n = {n}, total = {line_count}')

    
def save_orig_features_UCF():

    OUT_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\features\orig.jsonl'
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    model.eval()

    # register hook to get features
    activation = {}
    def get_activation(name):
        """Hook function to capture layer output"""
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    handle = model.avgpool.register_forward_hook(get_activation('features'))

    with open(UCF_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))    

    n = 0
    with open(UCF_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.0f}% is done', end='\r')
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            groups = record['groups']
            filename = record['filename']
            p = ucf101dm.construct_vid_path_from_full(record['filename'])
            video = ucf101dm.load_jpg_ucf101(p, n=0)
            p = model(video.permute(1,0,2,3)[None,:])
            features = activation['features']
            d = {}
            d['filename'] = filename
            d['feat'] = features[0,:,0,0,0].tolist()

            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(d) + '\n')

            n+=1

def save_orig_features_ssv2():

    GRP_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\groups\groups_0.001.jsonl'
    OUT_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\groups\features\orig.jsonl'
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    model = VJEPA2()
    model.eval()
    class_names = list(model.label2id.keys())

    # register hook to get features
    activation = {} 
    def get_activation(name):
        """Hook function to capture layer output"""
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    handle = model.model.pooler.self_attention_layers[2].mlp.fc2.register_forward_hook(get_activation('features'))

    data_dir = Path(CONST.SSV2_PATH)

    line_count = 0
    
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))   

    n=0
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.0f}% is done', end='\r')
            n+=1
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            groups = record['groups']
            filename = record['filename']
            splt = filename.split('/')
            video_path = Path.joinpath(data_dir, splt[-2], splt[-1])
            model.predict_from_path(video_path)

            features = activation['features']
            d = {}
            d['filename'] = splt[-2] + '/' + splt[-1]
            d['feat'] = features.mean(dim=1)[0].tolist()

            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(d) + '\n')
   
def save_grp_features_UCF():
    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.01.jsonl'
    OUT_DIR = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\features'
    OUT_PATH = os.path.join(OUT_DIR, "grp_" + os.path.basename(GRP_PATH).split('_')[-1])
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    model.eval()

    # register hook to get features
    activation = {}
    def get_activation(name):
        """Hook function to capture layer output"""
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    handle = model.avgpool.register_forward_hook(get_activation('features'))

    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))    

    n = 0
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.0f}% is done', end='\r')
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            g  = record['groups']
            filename = record['filename']
            p = ucf101dm.construct_vid_path_from_full(record['filename'])
            video = ucf101dm.load_jpg_ucf101(p, n=0).permute(1,0,2,3)

            groups = {}
            for k in g:
                if 'frames' not in g[k]:
                    groups[int(k)] = []
                else:
                    groups[int(k)] = g[k]['frames']

            vid_g = func.create_grouped_video(video, groups)

            p = model(vid_g[None,:])
            features = activation['features']
            d = {}
            d['filename'] = filename
            d['feat'] = features[0,:,0,0,0].tolist()

            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(d) + '\n')

            n+=1

def save_group_features_ssv2(thre):

    GRP_PATH = rf'C:\Users\lahir\Downloads\ssv2_analysis\groups\groups_{thre}.jsonl'
    OUT_DIR = r'C:\Users\lahir\Downloads\ssv2_analysis\groups\features'
    OUT_PATH = os.path.join(OUT_DIR, "grp_" + os.path.basename(GRP_PATH).split('_')[-1])
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    model = VJEPA2()
    model.eval()
    class_names = list(model.label2id.keys())

    # register hook to get features
    activation = {}
    def get_activation(name):
        """Hook function to capture layer output"""
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    handle = model.model.pooler.self_attention_layers[2].mlp.fc2.register_forward_hook(get_activation('features'))

    data_dir = Path(CONST.SSV2_PATH)

    line_count = 0
    
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))   

    n=0
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.0f}% is done', end='\r')
            n+=1
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            g = record['groups']
            filename = record['filename']
            splt = filename.split('/')

            vid = model.video_from_path(Path.joinpath(data_dir, splt[-2], splt[-1]))['pixel_values_videos'][0,:]
            groups = {}
            for k in g:
                if 'frames' not in g[k]:
                    groups[int(k)] = []
                else:
                    groups[int(k)] = g[k]['frames']

            vid_g = func.create_grouped_video(vid.permute(1,0,2,3), groups)
            model(vid_g[None,:])

            features = activation['features']
            d = {}
            d['filename'] = splt[-2] + '/' + splt[-1]
            d['feat'] = features.mean(dim=1)[0].tolist()

            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(d) + '\n')


def freeze_grp_feat(model, video, groups, FILL, device):

    # register hook to get features
    activation = {}
    def get_activation(name):
        """Hook function to capture layer output"""
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    handle = model.avgpool.register_forward_hook(get_activation('features'))

    # remove a randomly choosen group
    sel_idx = random.sample(list(range(len(groups.keys()))),1)[0]
    mask = [True]*len(groups.keys())
    mask[sel_idx] = False

    if FILL=='past':
        groups_filled = func.past_fill_all(mask, groups)
    elif FILL=='future':
        groups_filled = func.future_fill_all(mask, groups)
    elif FILL=='hybrid_mid':
        groups_filled = func.hybrid_fill_all(mask, groups, 'middle')
    elif FILL=='hybrid_random':
        groups_filled = func.hybrid_fill_all(mask, groups, 'random')
    if FILL!='late_sum':
        vid_g = func.create_grouped_video(video.permute(1,0,2,3), groups_filled).to(device)
        model(vid_g[None,:])
        feat = activation['features'][0,:,0,0,0]
    else:
        groups_filled = func.past_fill_all(mask, groups)
        vid_g = func.create_grouped_video(video.permute(1,0,2,3), groups_filled).to(device)
        model(vid_g[None,:])
        feat_past = activation['features'][0,:,0,0,0]

        groups_filled = func.future_fill_all(mask, groups)
        vid_g = func.create_grouped_video(video.permute(1,0,2,3), groups_filled).to(device)
        model(vid_g[None,:])
        feat_future = activation['features'][0,:,0,0,0]

        feat = (feat_past+feat_future)*0.5

    
    return feat

def zero_grp_feat(model, video, groups, device):

    # register hook to get features
    activation = {}
    def get_activation(name):
        """Hook function to capture layer output"""
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    handle = model.avgpool.register_forward_hook(get_activation('features'))

    # remove a randomly choosen group
    sel_idx = random.sample(list(range(len(groups.keys()))),1)[0]
    sel_key = list(groups.keys())[sel_idx]
    zero_idx = [sel_key] + groups[sel_key] 

    vid_g = func.create_grouped_video(video.permute(1,0,2,3), groups).to(device)
    for i in zero_idx:
        vid_g[:,i,:] = 0
    model(vid_g[None,:])
    feat = activation['features'][0,:,0,0,0]
        
    return feat

def tmp_freeze_grps_UCF101(FILL):
    import torch
    import func
    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f = f'{FILL}.jsonl'
    OUT_PATH = os.path.join(r'C:\Users\lahir\Downloads\UCF101\analysis\groups\freezing',f)
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
    
    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    model.eval()
    model.to(device)

    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f)) 
    
    n = 0
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.0f}% is done', end='\r')
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n+=1
            # if not record['filename']=='v_ApplyEyeMakeup_g01_c01':continue
            g = record['groups']
            groups = {}
            for k in g:
                if 'frames' not in g[k]:
                    groups[int(k)] = []
                else:
                    groups[int(k)] = g[k]['frames']
            p = ucf101dm.construct_vid_path_from_full(record['filename'])
            video = ucf101dm.load_jpg_ucf101(p, n=0)
            
            if FILL=='zero':
                feat = zero_grp_feat(model, video, groups, device)
            else:
                feat = freeze_grp_feat(model, video, groups, FILL, device)
            
            d = {}
            d['filename'] = record['filename']
            d['feat'] = feat.tolist()

            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(d) + '\n')

if __name__ == '__main__':
    # tmp_freeze_grps_UCF101('zero')
    save_group_features_ssv2(0.0001)
    save_group_features_ssv2(0.0005)
    save_group_features_ssv2(0.001)
    save_group_features_ssv2(0.005)
    save_group_features_ssv2(0.01)
    save_orig_features_ssv2()