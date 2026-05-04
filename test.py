import shap
import numpy as np
import func
import json
import torch    

ucf101dm = func.UCF101_data_model()
model = ucf101dm.model
UCF_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'
UCF_AVG_PRED = torch.tensor([-4.4024e-01, -4.0674e-02,  7.1152e-01, -3.1609e-01,  3.2655e-01,
         1.7831e-01, -5.8854e-02,  1.8917e-01, -2.1221e-01, -3.4419e-02,
        -5.6707e-01,  3.1817e-01, -5.1251e-01,  7.8728e-02,  3.8241e-01,
         4.4927e-01,  4.1014e-01, -4.8625e-01, -1.0066e+00, -5.5163e-01,
         3.9891e-01,  4.5372e-02,  3.8610e-01,  3.4739e-01,  9.7761e-02,
        -2.0676e-01, -9.5986e-02,  1.1389e-01,  6.2678e-01,  7.2580e-02,
        -7.7313e-01, -1.0116e+00, -1.3274e-02, -8.0600e-02,  2.9928e-01,
         8.2731e-02,  3.4950e-02,  1.0807e+00, -2.0324e-01, -3.5794e-01,
        -4.5873e-01, -7.5685e-01,  5.8560e-01, -4.4144e-01, -1.4889e-01,
        -3.4966e-01, -5.2623e-02,  7.0776e-01, -1.3836e+00,  4.2931e-02,
        -2.8714e-01,  7.1029e-01,  1.3096e-01,  3.8652e-01, -2.7135e-01,
         3.3671e-01,  2.8045e-01,  8.6642e-01,  2.7443e-01,  8.2717e-01,
         3.5757e-01,  7.5802e-01, -5.2487e-01, -4.5703e-01,  5.8566e-01,
        -2.6089e-02,  6.8762e-02,  2.0057e-02, -2.0009e-01,  1.8733e-03,
         4.6717e-01,  4.6903e-01, -1.2525e+00, -2.2765e-01,  8.1686e-01,
        -7.7576e-02, -1.9280e-01,  1.7929e-01,  1.3325e+00, -5.9062e-01,
        -1.5585e+00, -1.9866e+00, -1.3540e-01,  2.2269e-01, -3.5128e-01,
         3.8874e-01,  1.1447e+00, -6.0240e-01, -6.0129e-01,  5.1854e-01,
         3.8325e-01, -2.7351e-01,  2.6948e-01, -8.2457e-01,  5.1184e-01,
        -5.6957e-02,  2.5863e-01, -9.5546e-01,  7.0229e-01,  5.4221e-01,
         2.4085e-01])

def deep_copy_dict(dict):
    new_dict = {}
    for k in dict:
        new_dict[k] = dict[k].copy()
    return new_dict

'''
    modifies the group dict inplace
    use:     key_to_fill = 1
    future_fill(key_to_fill, mask, groups)
'''
def future_fill(fill_key, mask, groups):
    # #create a deep copy of the dict
    # new_groups = {}
    # for k in groups:
    #     new_groups[k] = groups[k].copy()
    ord_keys = sorted(mask.keys())
    l = [k for k in ord_keys if (k>fill_key and mask[k])]
    if len(l)==0:
        return -1
    first_true_key = l[0]
    groups[first_true_key].extend(list(set(groups[fill_key]+[fill_key])))
    groups.pop(fill_key)

def past_fill(fill_key, mask, groups):
    ord_keys = sorted(mask.keys(),reverse=True)
    l = [k for k in ord_keys if (k<fill_key and mask[k])]
    if len(l)==0:
        return -1
    first_true_key = l[0]
    groups[first_true_key].extend(list(set(groups[fill_key]+[fill_key])))
    groups.pop(fill_key)

def future_fill_all(mask, groups):
    groups = deep_copy_dict(groups)
    ord_keys = sorted(mask.keys())
    for k in ord_keys[:-1]:
        if not mask[k]:
            ret = future_fill(k, mask, groups)
            if ret ==-1:
                past_fill(k, mask, groups)
    if not mask[ord_keys[-1]]:
        past_fill(ord_keys[-1], mask, groups)
    return groups

def past_fill_all(mask, groups):
    groups = deep_copy_dict(groups)
    ord_keys = sorted(mask.keys())
    if not mask[ord_keys[0]]:
        future_fill(ord_keys[0], mask, groups)
    for k in ord_keys[1:]:
        if not mask[k]:
            ret = past_fill(k, mask, groups)
            if ret ==-1:
                future_fill(k, mask, groups)
    return groups

def get_video():
    with open(UCF_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            filename = record['filename']
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
            return video, groups
        

def predict_with_mask(mask):
    preds = torch.empty(0)
    for i, m in enumerate(mask):
        if sum(m)==0:
            # mean_frame = torch.mean(grouped_video, dim=(0,1))
            # masked = mean_frame[None,None,:].repeat(16,3,1,1)
            # p = model(masked[None,:].permute(0,2,1,3,4))
            preds = torch.concat([preds,UCF_AVG_PRED[None,:]],axis=0)
        else:
            masked = grouped_video.clone()
            ord_keys = sorted(list(groups.keys()))
            m_ = {}
            for i,k in enumerate(ord_keys):
                m_[int(k)] = bool(m[i])
            g = past_fill_all(m_, groups)
            vid_g = func.create_grouped_video(masked.permute(1,0,2,3), g)
            vid_g = vid_g.permute(1,0,2,3)
            p = model(vid_g.permute(1,0,2,3)[None,:])
            preds = torch.concat([preds,p],axis=0)

    return preds.detach().numpy()

video, groups = get_video()
grouped_video = func.create_grouped_video(video.permute(1,0,2,3), groups).permute(1,0,2,3)
NUM_GROUPS = len(groups)
background = np.zeros((1, NUM_GROUPS))

masker = shap.maskers.Independent(
    data=background,
    max_samples=2 ** NUM_GROUPS   # allow full exact enumeration
)
explainer = shap.Explainer(
    model=predict_with_mask,
    masker=masker,
    algorithm="exact"             # force exact Shapley computation
)
test_instance = np.ones((1, NUM_GROUPS)) 
shap_values = explainer(test_instance)


sv = shap_values.values[0,:]
bv = shap_values.base_values[0,:]
p  = model(grouped_video.permute(1,0,2,3)[None,:])[0,:].detach().numpy()

sv = np.sum(sv,axis=0)

abs(p - bv - sv).mean()



pass