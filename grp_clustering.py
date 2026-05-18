import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import StandardScaler
import os
import json

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

if __name__ == '__main__':
    cluster_features()