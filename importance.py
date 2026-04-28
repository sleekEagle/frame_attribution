import shap
import numpy as np
import func
import json

UCF_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'


class CalcSHAP:
    def __init__(self):
        self.explainer = shap.explainers.Exact(self.predict, self.custom_masker)
        self.n_masks = 0

    def predict(self, x):
        print('in predict')
        return np.random.rand(x.shape[0], 3)  # Dummy prediction function for binary classification

    def custom_masker(self, mask, x):
        self.n_masks += 1
        #create the new video with the given mask
        self.groups
        mask
        return (x * mask).reshape(1, len(x))
    
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
    ex = CalcSHAP()

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
    first_true_key = [k for k in ord_keys if (k>fill_key and mask[k])][0]
    groups[first_true_key].extend(groups[fill_key])
    groups.pop(fill_key)

def past_fill(fill_key, mask, groups):
    # #create a deep copy of the dict
    # new_groups = {}
    # for k in groups:
    #     new_groups[k] = groups[k].copy()
    ord_keys = sorted(mask.keys(),reverse=True)
    first_true_key = [k for k in ord_keys if (k<fill_key and mask[k])][0]
    groups[first_true_key].extend(groups[fill_key])
    groups.pop(fill_key)

def test():
    groups = {1: [0, 2], 3: [3], 4: [5,6,7], 10: [8,9], 11:[12,13,14], 16:[15,17]}
    m = [False, False, True, False, True, False]
    ord_keys = sorted(groups.keys())
    mask = {}
    for i,k in enumerate(ord_keys):
        mask[k] = m[i]

    key_to_fill = 1
    future_fill(key_to_fill, mask, groups)



        




    pass






if __name__ == "__main__":
    test()