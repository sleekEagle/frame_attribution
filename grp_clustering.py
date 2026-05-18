import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import StandardScaler
import os
import json
import random

UCF_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'
PLOT_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\plots\grouping\grp_clustering'

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div, rel_entr
from scipy.special import softmax

def cluster_features():

    DIR = r'C:\Users\lahir\Downloads\UCF101\analysis\features'
    thre = os.path.basename(UCF_PATH).split('_')[1][:-6]
    GRP_PATH = os.path.join(DIR, f'grp_{thre}.jsonl')
    ORIG_PATH = os.path.join(DIR, "orig.jsonl")
    scaler = StandardScaler()

    with open(ORIG_PATH, 'r') as f:
        orig_feat, org_f_names = [],[]
        for line in f:
            orig_feat.append(json.loads(line)['feat'])
            org_f_names.append(json.loads(line)['filename'])
    with open(GRP_PATH, 'r') as f:
        grp_feat, grp_f_names = [],[]
        for line in f:
            grp_feat.append(json.loads(line)['feat'])
            grp_f_names.append(json.loads(line)['filename'])

    for i in range(len(org_f_names)):
        assert org_f_names[i] == grp_f_names[i], 'file name order does not match!'

    orig_feat, grp_feat = np.array(orig_feat).astype(np.float32), np.array(grp_feat).astype(np.float32)

    #get L2 distance between the clusters
    avg_dist = np.mean(np.sqrt(np.sum((orig_feat - grp_feat)**2, axis=1)))
    print(f'Distance between the two clusters: {avg_dist}')
    # get KL Divergence between the two clusters
    probs_orig = softmax(orig_feat, axis=1)  # Shape: (100, 2048)
    probs_grp = softmax(grp_feat, axis=1)  # Shape: (100, 2048)
    epsilon = 1e-10
    probs_orig = np.clip(probs_orig, epsilon, 1.0)
    probs_grp = np.clip(probs_grp, epsilon, 1.0)
    kl_scores = np.sum(rel_entr(probs_orig, probs_grp), axis=1)
    avg_kl = np.mean(kl_scores)
    print(f'Distance KL-Div between the two clusters: {avg_kl}')

    with open(os.path.join(PLOT_PATH, 'results.txt'), 'a') as file:
        file.write(f'Threshold:{thre}, L2: {avg_dist}, KL-Div: {avg_kl}\n')


    all_features = np.vstack([orig_feat, grp_feat])
    # scaler = StandardScaler()
    # all_features = scaler.fit_transform(all_features)
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    
    orig_pca = features_2d[:len(orig_feat)]
    grp_pca = features_2d[len(grp_feat):]



    plt.figure(figsize=(10, 8))

    # Plot first cluster in blue
    plt.scatter(orig_pca[:, 0], orig_pca[:, 1], 
                c='blue', label='Original', alpha=0.6, s=10, linewidth=0)

    # Plot second cluster in red
    plt.scatter(grp_pca[:, 0], grp_pca[:, 1], 
                c='red', label='Grouped', alpha=0.6, s=10, linewidth=0)

    plt.title(f'2D Clusters of Original and Groups with Threshold = {thre}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_PATH, f'threshold_{thre}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def tmp_freeze_grps():
    import torch
    import func
    FILL = 'future'
    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f = f'{FILL}.jsonl'
    OUT_PATH = os.path.join(r'C:\Users\lahir\Downloads\UCF101\analysis\features\tmp_freeze',f)
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
    
    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    model.eval()
    model.to(device)
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
            g = record['groups']
            groups = {}
            for k in g:
                if 'frames' not in g[k]:
                    groups[int(k)] = []
                else:
                    groups[int(k)] = g[k]['frames']
            p = ucf101dm.construct_vid_path_from_full(record['filename'])
            video = ucf101dm.load_jpg_ucf101(p, n=0)

            #remove a randomly choosen group
 
            sel_idx = random.sample(list(range(len(groups.keys()))),1)[0]
            mask = [True]*len(groups.keys())
            mask[sel_idx] = False

            if FILL=='past':
                groups_filled = func.past_fill_all(mask, groups)
                vid_g = func.create_grouped_video(video.permute(1,0,2,3), groups_filled).to(device)
                p = model(vid_g[None,:])
                feat = activation['features'][0,:,0,0,0]
            elif FILL=='future':
                groups_filled = func.future_fill_all(mask, groups)
                vid_g = func.create_grouped_video(video.permute(1,0,2,3), groups_filled).to(device)
                p = model(vid_g[None,:])
                feat = activation['features'][0,:,0,0,0]
            elif FILL=='comb':
                groups_filled = func.past_fill_all(mask, groups)
                vid_g = func.create_grouped_video(video.permute(1,0,2,3), groups_filled).to(device)
                p = model(vid_g[None,:])
                feat_past = activation['features'][0,:,0,0,0]

                groups_filled = func.future_fill_all(mask, groups)
                vid_g = func.create_grouped_video(video.permute(1,0,2,3), groups_filled).to(device)
                p = model(vid_g[None,:])
                feat_future = activation['features'][0,:,0,0,0]

                feat = (feat_past+feat_future)*0.5
            
            d = {}
            d['filename'] = record['filename']
            d['feat'] = feat.tolist()

            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(d) + '\n')


            n+=1


def cluster_frozen():
    import pandas as pd
    df = pd.DataFrame(columns=['filename', 'orig', 'future', 'past', 'comb'])

    ORIG_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\features\orig.jsonl'
    FUTURE_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\features\tmp_freeze\future.jsonl'
    PAST_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\features\tmp_freeze\past.jsonl'
    COMB_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\features\tmp_freeze\comb.jsonl'

    #read original features
    def get_data(path):
        d = {}
        with open(path, 'r') as f:
            for line in f:
                d[json.loads(line)['filename']] = json.loads(line)['feat']
        return d
    
    d_orig = get_data(ORIG_PATH)
    d_future = get_data(FUTURE_PATH)
    d_past = get_data(PAST_PATH)
    d_comb = get_data(COMB_PATH)

    rows_list = []
    for k in d_orig:
        row = {'filename': k, 
               'orig': np.array(d_orig[k]), 
               'future': np.array(d_future[k]), 
               'past': np.array(d_past[k]), 
               'comb': np.array(d_comb[k])}
        rows_list.append(row)
    df = pd.concat([df, pd.DataFrame(rows_list)], ignore_index=True)

    o = np.stack(df['orig'].values)  # Shape: (n_rows, 128)
    f = np.stack(df['future'].values)  # Shape: (n_rows, 128)
    p = np.stack(df['past'].values) 
    c = np.stack(df['comb'].values) 

    # Calculate L2 distances for all rows at once
    df['diff_future'] = np.linalg.norm(o - f, axis=1)
    df['diff_past'] = np.linalg.norm(o - p, axis=1)
    df['diff_comb'] = np.linalg.norm(o - c, axis=1)

    probs_orig = softmax(o, axis=1)  
    probs_f = softmax(f, axis=1) 
    probs_p = softmax(p, axis=1)
    probs_c = softmax(c, axis=1)

    epsilon = 1e-10
    probs_orig = np.clip(probs_orig, epsilon, 1.0)
    probs_f = np.clip(probs_f, epsilon, 1.0)
    probs_p = np.clip(probs_p, epsilon, 1.0)
    probs_c = np.clip(probs_c, epsilon, 1.0)
    kl_f = np.sum(rel_entr(probs_orig, probs_f), axis=1)
    kl_p = np.sum(rel_entr(probs_orig, probs_p), axis=1)
    kl_c = np.sum(rel_entr(probs_orig, probs_c), axis=1)
    
    df['entr_future'] = kl_f
    df['entr_past'] = kl_p
    df['entr_comb'] = kl_c

    pass



if __name__ == '__main__':
    cluster_frozen()