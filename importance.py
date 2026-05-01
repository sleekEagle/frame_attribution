import shap
import numpy as np
import func
import json
import torch

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

class CalcSHAP:
    def __init__(self, model):
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.explainer = shap.explainers.Exact(self.predict, self.custom_masker)
        self.n_masks = 0
        self.BACKGROUND = "PAST"
        self.avg_pred = UCF_AVG_PRED

    def predict(self, x):

        zero_idx = [i for i in range(x.shape[0]) if x[i,:].sum()==0.0]
        nonzero_idx = [i for i in range(x.shape[0]) if i not in zero_idx]
        assert len(zero_idx) <=1, "there cannot be more than 1 all-zero masks"

        x_nz = torch.tensor(x[nonzero_idx], dtype=torch.float32).cuda()

        pred_t = torch.zeros(x.shape[0], self.avg_pred.shape[0])
        with torch.no_grad():
            pred_nz =  self.model(x_nz.permute(0,2,1,3,4)).cpu()
            pred_t[nonzero_idx,:] = pred_nz
            if len(zero_idx)==1:
                pred_z = self.avg_pred[None,:]
                pred_t[zero_idx,:] = pred_z
       
        return pred_t.numpy()
        # return np.random.rand(x.shape[0], 3)  # Dummy prediction function for binary classification

    def custom_masker(self, m, grp_feat):
        self.n_masks += 1
        ord_keys = sorted(grp_feat.tolist())
        mask = {}
        for i,k in enumerate(ord_keys):
            mask[int(k)] = bool(m[i])

        # handle all-false mask seperately
        if m.sum() == 0:
            vid_g = torch.zeros_like(self.video)
        else:
            g = past_fill_all(mask, self.groups)
            vid_g = func.create_grouped_video(self.video.permute(1,0,2,3), g)
            vid_g = vid_g.permute(1,0,2,3)
        
        return vid_g[None,:]
    
    def explain(self, video, groups):
        self.n_masks = 0
        self.video = video
        self.groups = groups
        grp_features = np.array([list(groups.keys())])
        e = self.explainer(grp_features)
        return e





# # Train a model (example with binary classification)
# X = np.random.rand(1, 16)

# explainer = shap.explainers.Exact(predict, custom_masker)
# shap_values = explainer(X)  # Explain the first sample

# # Get just the explanations for the positive class
# shap_values = shap_values[..., 1]

# print(f"Total masks used: {masks}")
# print(f"Expected maximum: {2**5} = 32 masks per sample")
# print(f"Per sample average: {masks / 1:.1f}")
# pass


def calc_shap():
    #****************************************************************************
    # the model and the data loader
    #****************************************************************************
    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    inference_loader = ucf101dm.inference_loader
    inference_class_names = ucf101dm.inference_class_names
    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k
    #****************************************************************************

    #*************************************************************************
    # initialize shap model 
    #*************************************************************************
    ex = CalcSHAP(model)

    #read groups
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

            shap_values = ex.explain(video, groups)
            pass

def test():
    groups = {1: [0, 2, 3], 4: [], 8: [6,7], 10: [], 13:[14], 16:[15,17,18]}
    m = [False, False, False, False, False, False]
    ord_keys = sorted(groups.keys())
    mask = {}
    for i,k in enumerate(ord_keys):
        mask[k] = m[i]

    # key_to_fill = 1
    # future_fill(key_to_fill, mask, groups)

    past_fill_all(mask, groups)
    print(groups)
    print(sum([len(groups[k]) for k in groups]))



        

    pass






if __name__ == "__main__":
    calc_shap()