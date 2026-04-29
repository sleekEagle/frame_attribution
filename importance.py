import shap
import numpy as np
import func
import json
import torch

UCF_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'

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
        self.avg_pred = None

    def predict(self, x):

        zero_idx = [i for i in range(x.shape[0]) if x[i,:].sum()==0.0]
        assert len(zero_idx)==1, "There should be exactly one all-zero mask in the batch"
        nonzero_idx = [i for i in range(x.shape[0]) if i not in zero_idx]

        x_nz = torch.tensor(x[nonzero_idx], dtype=torch.float32).cuda()
        x_z = x[zero_idx]

        with torch.no_grad(), torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred_nz =  self.model(x_nz.permute(0,2,1,3,4)).cpu().numpy()
            pred_z = self.avg_pred
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
        self.video = video
        self.groups = groups
        grp_features = np.array([list(groups.keys())])
        self.explainer(grp_features)

        pass





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
                f = g[k]['frames']
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