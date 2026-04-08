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

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Overlaid histograms
    plt.hist(in_d_l, bins=30, alpha=0.5, label='in_diff')
    plt.hist(out_d_l, bins=30, alpha=0.5, label='out_diff')
    plt.legend()
    plt.show(block=False)

    sns.boxplot(data=[in_d_l, out_d_l])

    diff_l = [out_d_l[i]-in_d_l[i] for i in range(len(in_d_l))]
    plt.hist(diff_l, bins=30, alpha=0.5, label='diff_l')
    plt.title('Histogram of out_diff - mean_in_diff')

    # see if there is a trend in the in_diff_values and how does to compare to the out_diff_values.
    max_len = max([len(v) for v in in_grp])+1
    value_l = [[] for _ in range(max_len)]
    for i in range(len(in_grp)):
        if len(in_grp[i])<=1: continue
        in_v = in_grp[i]
        out_v = out_grp[i]
        v = np.array(in_v + [out_v])
        # v = v - v[0] #min shift
        #min max scale
        # v = ((v - np.min(v)) / (np.max(v) - np.min(v)))
        v = v.tolist()

        for j in range(len(v)-1):
            value_l[j].append(v[j])
        value_l[-1].append(v[-1])
    
    # std_l = [float(np.std(v)) for v in value_l]
    # mean_l = [float(np.mean(v)) for v in value_l]

    #violin plot
    labels = list(range(1, max_len+1))
    plt.figure(figsize=(10, 6))
    plt.violinplot(value_l, positions=range(1, len(labels)+1))
    plt.xticks(range(1, len(labels)+1), labels)
    plt.ylabel('Values')
    plt.title('Violin Plot ith per change values')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == '__main__':
    UCF101_minchange()