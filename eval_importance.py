import shap
import numpy as np
import func
import json
import torch    
import os
from pathlib import Path
import CONST

#get all pred original logits from the group log file
def get_orig_logits(PATH):
    d = {}
    with open(PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)            
            d[record['filename']] = record['original_logit']
    return d

class EvalLogits:
    def __init__(self, full_video, model, groups, ol, cls_idx):
        self.full_video = full_video
        self.model = model
        self.groups = groups
        self.ol = ol
        self.cls_idx = cls_idx

    def eval(self, idx_order):
        logits = [self.ol]
        mask = [1]*len(self.groups)
        for idx in idx_order:
            mask[idx]=0
            print(mask)
            if sum(mask) == 0:
                logits.append(CONST.UCF_AVG_PRED[self.cls_idx].item())
                continue

            g = func.past_fill_all(mask, self.groups)
            vid_g = func.create_grouped_video(self.full_video.permute(1,0,2,3), g)
            with torch.no_grad():
                p = self.model(vid_g[None,:])
                l = p[0,self.cls_idx].item()
                logits.append(l)
        return logits

def eval_UCF101():
    IMP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\exactSHAP_0.001.jsonl'
    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'

    orig_logits = get_orig_logits(GRP_PATH)

    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k

    with open(IMP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))

    with open(IMP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            class_labels
            cls_idx = class_labels[record['filename'].split('_')[1].lower()]
            ol = orig_logits[record['filename']]

            p = ucf101dm.construct_vid_path_from_full(record['filename'])
            video = ucf101dm.load_jpg_ucf101(p, n=0)
            p = model(video.permute(1,0,2,3)[None,:])
            l = p[0,cls_idx]
            sv = record['shapley_values'][0]

            # sanity checks
            assert abs(ol-l.item()) < 1e-5, 'the prediction logit does not match the original prediction logit'
            assert record['difference'] < 1e-5, 'The exactly shapley value difference is too large!'
            assert len(record['groups']) == len(sv) , 'Number of shapley values do not match!'

            sv_c = [s[cls_idx] for s in sv]
            sort_idx = [int(i) for i in np.argsort(sv_c)]

            # let groups have integer keys
            groups = {}
            for k in record['groups']:
                groups[int(k)] = record['groups'][k]

            el = EvalLogits(video, model, groups, ol, cls_idx)

            
            el.eval(sort_idx)
            pass
            







            pass

    pass

if __name__ == "__main__":
    eval_UCF101()
