import shap
import numpy as np
import func
import json
import torch    
import os
from pathlib import Path
import CONST
import random
from scipy.stats import entropy
import torch.nn.functional as F
from scipy.integrate import trapezoid
import math


#get all pred original logits from the group log file
def get_orig_logits(PATH):
    d = {}
    with open(PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)            
            d[record['filename']] = {'stats': record['all_grp_stats'], 'correct': record['correct']}
    return d

class EvalLogits:
    def __init__(self, full_video, model, groups, orig_stat, cls_idx, fill_type, show_mask=False):
        self.full_video = full_video
        self.model = model
        self.groups = groups
        self.orig_stat = orig_stat
        self.cls_idx = cls_idx
        self.show_mask = show_mask
        self.fill_type = fill_type
        self.N_CLS = 101

        #prepare the all and none metrics
        all_metrics, none_metrics = {},{}

        # none_metrics['entropy'] = float(entropy(F.softmax(CONST.UCF_AVG_PRED,dim=0)))
        # none_metrics['logit'] = float(CONST.UCF_AVG_PRED[cls_idx])
        # for m in [1,3,5]:
        #     none_metrics[f'margin_{m}'] = float(func.get_margin(CONST.UCF_AVG_PRED, cls_idx=cls_idx, k=m))
        none_metrics['entropy'] = math.log(self.N_CLS)
        none_metrics['logit'] = 1/self.N_CLS
        for m in [1,3,5]:
            none_metrics[f'margin_{m}'] = 0.0

        logits = torch.tensor(orig_stat['stats']['logits'])
        all_metrics['entropy'] = orig_stat['stats']['entropy']
        all_metrics['logit'] = float(logits[cls_idx])
        for m in [1,3,5]:
            all_metrics[f'margin_{m}'] = float(func.get_margin(logits, cls_idx=cls_idx, k=m))

        self.all_metrics, self.none_metrics = all_metrics, none_metrics

    def eval_remove(self, idx_order):
        metrics = {
            'entropy': [0],
            'margin1': [1],
            'margin3': [1],
            'margin5': [1],
            'logit': [1]
        }
        mask = [1]*len(self.groups)
        if self.show_mask:
            print(mask)
        for idx in idx_order:
            mask[idx] = 0
            if self.show_mask :
                print(mask)
            if sum(mask) == 0:
                metrics['entropy'].append(1)
                metrics['margin1'].append(0)
                metrics['margin3'].append(0)
                metrics['margin5'].append(0)
                metrics['logit'].append(0)
                continue
            if self.fill_type == 'past':
                g = [func.past_fill_all(mask, self.groups)]
            elif self.fill_type == 'future':
                g = [func.future_fill_all(mask, self.groups)]
            elif self.fill_type == 'middle':
                g = [func.hybrid_fill_all(mask, self.groups, 'middle')]
            elif self.fill_type=='random':
                g = [func.hybrid_fill_all(mask, self.groups, 'random')]
            elif self.fill_type=='late':
                g = [func.past_fill_all(mask, self.groups),
                         func.future_fill_all(mask, self.groups)]

            avg_ = {
                'entropy' : 0,
                'logit': 0,
                'margin1': 0,
                'margin3': 0,
                'margin5': 0,
            }
            n = 0
            for g_ in g:
                vid_g = func.create_grouped_video(self.full_video, g_)
                stat = func.get_pred_stats(self.model, vid_g)
                # assert (stat['entropy'] - self.all_metrics['entropy'])/(self.none_metrics['entropy'] - self.all_metrics['entropy'])>0,'error'
                avg_['entropy'] += (stat['entropy'] - self.all_metrics['entropy'])/(self.none_metrics['entropy'] - self.all_metrics['entropy'])
                avg_['logit'] += (stat['logits'][self.cls_idx] - self.none_metrics['logit'])/(self.all_metrics['logit'] - self.none_metrics['logit'])
                for m in [1,3,5]:
                    avg_[f'margin{m}'] += (stat[f'margin_{m}'] - self.none_metrics[f'margin_{m}'])/(self.all_metrics[f'margin_{m}'] - self.none_metrics[f'margin_{m}'])
                n += 1
            for k in avg_:
                metrics[k].append(avg_[k]/n)

        auc = {}
        x = np.linspace(0, 1, len(metrics['entropy']))
        # assert float(trapezoid(metrics['entropy'], x)) >=0 ,'entropy auc is negative'
        auc['entropy'] = float(trapezoid(metrics['entropy'], x))
        auc['logit'] = float(trapezoid(metrics['logit'], x))
        for m in [1,3,5]:
            # assert float(trapezoid(metrics[f'margin{m}'], x)) >=0 ,f'margin{m} auc is negative'
            auc[f'margin{m}'] = float(trapezoid(metrics[f'margin{m}'], x))


        # mask = [1,0,0]
        # # mask[0]=0
        # g_ = func.past_fill_all(mask, self.groups)
        # vid_g = func.create_grouped_video(self.full_video, g_)
        # stat = func.get_pred_stats(self.model, vid_g)
        
  
        return {'list': metrics, 'auc': auc}
    
    
    def eval_add(self, idx_order):
        metrics = {
            'entropy': [1],
            'margin1': [0],
            'margin3': [0],
            'margin5': [0],
            'logit': [0]
        }
        mask = [0]*len(self.groups)
        if self.show_mask:
            print(mask)
        for idx in idx_order:
            mask[idx] = 1
            if self.show_mask :
                print(mask)
            if sum(mask) == 0:
                metrics['entropy'].append(0)
                metrics['margin1'].append(1)
                metrics['margin3'].append(1)
                metrics['margin5'].append(1)
                metrics['logit'].append(1)
                continue
            if self.fill_type == 'past':
                g = [func.past_fill_all(mask, self.groups)]
            elif self.fill_type == 'future':
                g = [func.future_fill_all(mask, self.groups)]
            elif self.fill_type == 'middle':
                g = [func.hybrid_fill_all(mask, self.groups, 'middle')]
            elif self.fill_type=='random':
                g = [func.hybrid_fill_all(mask, self.groups, 'random')]
            elif self.fill_type=='late':
                g = [func.past_fill_all(mask, self.groups),
                         func.future_fill_all(mask, self.groups)]

            avg_ = {
                'entropy' : 0,
                'logit': 0,
                'margin1': 0,
                'margin3': 0,
                'margin5': 0,
            }
            n = 0
            for g_ in g:
                vid_g = func.create_grouped_video(self.full_video, g_)
                stat = func.get_pred_stats(self.model, vid_g)
                avg_['entropy'] += (stat['entropy'] - self.all_metrics['entropy'])/(self.none_metrics['entropy'] - self.all_metrics['entropy'])
                avg_['logit'] += (stat['logits'][self.cls_idx] - self.none_metrics['logit'])/(self.all_metrics['logit'] - self.none_metrics['logit'])
                for m in [1,3,5]:
                    avg_[f'margin{m}'] += (stat[f'margin_{m}'] - self.none_metrics[f'margin_{m}'])/(self.all_metrics[f'margin_{m}'] - self.none_metrics[f'margin_{m}'])
                n += 1
            for k in avg_:
                metrics[k].append(avg_[k]/n)

        auc = {}
        x = np.linspace(0, 1, len(metrics['entropy']))
        auc['entropy'] = float(trapezoid(metrics['entropy'], x))
        auc['logit'] = float(trapezoid(metrics['logit'], x))
        for m in [1,3,5]:
            auc[f'margin{m}'] = float(trapezoid(metrics[f'margin{m}'], x))

        return {'list': metrics, 'auc': auc}

def eval_UCF101():
    FILL_TYPE = 'past' # past, future, middle, random, late
    IMP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\shap\exactSHAP_past_0.001.jsonl'
    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    OUT_PATH = rf'C:\Users\lahir\Downloads\UCF101\analysis\importance\eval\{FILL_TYPE}_0.001.jsonl'

    # GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.0001.jsonl'
    grp_stats = get_orig_logits(GRP_PATH)

    # idx = np.argmax(np.array(grp_data['v_BlowDryHair_g01_c02']['data']))
    # grp_data['v_BlowDryHair_g01_c02']['data'][idx]

    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k

    with open(IMP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))

    n=0
    with open(IMP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.1f}% is done.', end='\r')
            n+=1
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # if record['filename']!='v_BlowDryHair_g01_c02': continue
            cls_idx = class_labels[record['filename'].split('_')[1].lower()]
            orig_stat = grp_stats[record['filename']]

            # ol = grp_data[record['filename']]['data']

            p = ucf101dm.construct_vid_path_from_full(record['filename'])
            video = ucf101dm.load_jpg_ucf101(p, n=0).permute(1,0,2,3)

            # orig_stat = func.get_pred_stats(model, video)
            # l = orig_stat['logits'][cls_idx]
            sv = record['shapley_values'][0]

            # sanity checks
            # assert abs(ol[cls_idx]-l) < 1e-2, 'the prediction logit does not match the original prediction logit'
            # assert record['difference'] < 1e-2, 'The exactly shapley value difference is too large!'
            # assert len(record['groups']) == len(sv) , 'Number of shapley values do not match!'

            sv_c = [s[cls_idx] for s in sv]
            asc_idx = [int(i) for i in np.argsort(sv_c)]

            # let groups have integer keys
            groups = {}
            for k in record['groups']:
                groups[int(k)] = record['groups'][k]

            el = EvalLogits(video, model, groups, orig_stat, cls_idx, FILL_TYPE)

            results = {
                'filename': record['filename'],
                'rmv_asc': el.eval_remove(asc_idx),
                'rmv_dec': el.eval_remove(asc_idx[::-1]),
                'rmv_rand': el.eval_remove(random.sample(asc_idx, len(asc_idx))),
                'rmv_lr': el.eval_remove(sorted(asc_idx)),
                'rmv_rl': el.eval_remove(sorted(asc_idx)[::-1]),
                'add_asc': el.eval_add(asc_idx),
                'add_dec': el.eval_add(asc_idx[::-1]),
                'add_rand': el.eval_add(random.sample(asc_idx, len(asc_idx))),
                'add_lr': el.eval_add(sorted(asc_idx)),
                'add_rl': el.eval_add(sorted(asc_idx)[::-1])
            }

            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(results) + '\n')

def avg_stat_ucf():
    EVAL_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\importance\eval\past_0.001.jsonl'
    metrics = {
        'entropy':0,
        'logit':0,
        'margin1':0,
        'margin3':0,
        'margin5':0
    }
    d = {'rmv_asc':func.deep_copy_dict(metrics),
         'rmv_dec':func.deep_copy_dict(metrics) ,
         'rmv_rand':func.deep_copy_dict(metrics) ,
         'rmv_lr':func.deep_copy_dict(metrics) ,
         'rmv_rl':func.deep_copy_dict(metrics) ,
         'add_asc':func.deep_copy_dict(metrics) ,
         'add_dec':func.deep_copy_dict(metrics) ,
         'add_rand':func.deep_copy_dict(metrics) ,
         'add_lr':func.deep_copy_dict(metrics) ,
         'add_rl':func.deep_copy_dict(metrics)}
    
    n=0
    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d_ = json.loads(line)
            for k in set(d_.keys()) - {'filename'}:
                for m in d_[k]['auc']:
                    d[k][m] += d_[k]['auc'][m]
            n+=1
    for k in d:
        for m in d[k]:
            d[k][m]/=n
        
    metrics = ['entropy', 'logit', 'margin1', 'margin3', 'margin5']

    for m in metrics:
        print(f'\n{m} AUC: ')
        for k in d:
            s = f'{k} : {d[k][m]}'
            print(s)


def importance_correlation():
    pass

if __name__ == "__main__":
    importance_correlation()
