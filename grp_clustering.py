import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import StandardScaler
import os
import json
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div, rel_entr
from scipy.special import softmax

def cluster_features():
    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\features\grp_0.01.jsonl'
    PLOT_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\features\plots'
    DIR = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\features'
    thre = os.path.basename(GRP_PATH).split('_')[1][:-6]
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


def cluster_frozen():
    import pandas as pd
    df = pd.DataFrame(columns=['filename', 'orig', 'future', 'past', 'late_sum', 'hybrid_mid','hybrid_random'])

    ORIG_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\features\orig.jsonl'
    FUTURE_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\freezing\future.jsonl'
    PAST_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\freezing\past.jsonl'
    LATE_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\freezing\late_sum.jsonl'
    MID_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\freezing\hybrid_mid.jsonl'
    RAND_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\freezing\hybrid_random.jsonl'

    PLOT_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\freezing\plots'

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
    d_late = get_data(LATE_PATH)
    d_mid = get_data(MID_PATH)
    d_random = get_data(RAND_PATH)

    rows_list = []
    for k in d_orig:
        row = {'filename': k, 
               'orig': np.array(d_orig[k]), 
               'future': np.array(d_future[k]), 
               'past': np.array(d_past[k]), 
               'late': np.array(d_late[k]),
               'mid': np.array(d_mid[k]),
               'random': np.array(d_random[k])}
        rows_list.append(row)
    df = pd.concat([df, pd.DataFrame(rows_list)], ignore_index=True)

    original = np.stack(df['orig'].values)  # Shape: (n_rows, 128)
    future = np.stack(df['future'].values)  # Shape: (n_rows, 128)
    past = np.stack(df['past'].values) 
    late = np.stack(df['late'].values) 
    mid = np.stack(df['mid'].values) 
    random = np.stack(df['random'].values) 

    # Calculate L2 distances for all rows at once
    df['diff_future'] = np.linalg.norm(original - future, axis=1)
    df['diff_past'] = np.linalg.norm(original - past, axis=1)
    df['diff_late'] = np.linalg.norm(original - late, axis=1)
    df['diff_mid'] = np.linalg.norm(original - mid, axis=1)
    df['diff_random'] = np.linalg.norm(original - random, axis=1)

    probs_o = softmax(original, axis=1)  
    probs_f = softmax(future, axis=1) 
    probs_p = softmax(past, axis=1)
    probs_l = softmax(late, axis=1)
    probs_m = softmax(mid, axis=1)
    probs_r = softmax(random, axis=1)

    epsilon = 1e-10
    probs_orig = np.clip(probs_o, epsilon, 1.0)
    probs_f = np.clip(probs_f, epsilon, 1.0)
    probs_p = np.clip(probs_p, epsilon, 1.0)
    probs_l = np.clip(probs_l, epsilon, 1.0)
    probs_m = np.clip(probs_m, epsilon, 1.0)
    probs_r = np.clip(probs_r, epsilon, 1.0)

    kl_f = np.sum(rel_entr(probs_orig, probs_f), axis=1)
    kl_p = np.sum(rel_entr(probs_orig, probs_p), axis=1)
    kl_l = np.sum(rel_entr(probs_orig, probs_l), axis=1)
    kl_m = np.sum(rel_entr(probs_orig, probs_m), axis=1)
    kl_r = np.sum(rel_entr(probs_orig, probs_r), axis=1)
    
    df['kl_future'] = kl_f
    df['kl_past'] = kl_p
    df['kl_late'] = kl_l
    df['kl_mid'] = kl_m
    df['kl_random'] = kl_r

    
    def get_PCA(o,f):
        all_features = np.vstack([o, f])    
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)
        orig_pca = features_2d[:len(o)]
        f_pca = features_2d[len(f):]

        return orig_pca, f_pca
    
    

    PCA_metrics = {}
    for freeze_method in ['future','past','late','mid','random']:
        f = locals()[freeze_method]
        orig_pca, f_pca = get_PCA(original, f)

        plt.figure(figsize=(10, 8))
        plt.scatter(orig_pca[:, 0], orig_pca[:, 1], 
                    c='blue', label='Original', alpha=0.6, s=10, linewidth=0)
        plt.scatter(f_pca[:, 0], f_pca[:, 1], 
                    c='red', label=freeze_method, alpha=0.6, s=10, linewidth=0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOT_PATH, f'{freeze_method}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    


    pass

    print(df['kl_future'].mean())
    print(df['kl_past'].mean())
    print(df['kl_late'].mean())
    print(df['kl_mid'].mean())
    print(df['kl_random'].mean())

    print(df['diff_future'].mean())
    print(df['diff_past'].mean())
    print(df['diff_late'].mean())
    print(df['diff_mid'].mean())
    print(df['diff_random'].mean())

    # plot the clusters




# plot grouped feature distribution difference
# import matplotlib.pyplot as plt
# thr = [0.0001, 0.0005, 0.001, 0.005, 0.01]
# l2 = [8.446063041687012, 8.493795394897461, 8.548871994018555, 8.792335510253906, 9.067133903503418]
# kl = [0.029468843713402748, 0.029719814658164978, 0.030001485720276833, 0.031332600861787796, 0.03286220505833626]

# plt.plot(thr, l2, marker='o', markersize=4)
# plt.xlabel('Threshold')
# plt.ylabel('L2')
# plt.grid(True, alpha=0.3)
# plt.savefig(os.path.join(r'C:\Users\lahir\Downloads\UCF101\analysis\groups\features\plots\L2.png'), dpi=300, bbox_inches='tight')


# plt.plot(thr, kl, marker='o', markersize=4)
# plt.xlabel('Threshold')
# plt.ylabel('KL-Div')
# plt.grid(True, alpha=0.3)
# plt.savefig(os.path.join(r'C:\Users\lahir\Downloads\UCF101\analysis\groups\features\plots\KL.png'), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    cluster_frozen()