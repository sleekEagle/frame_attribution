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
import pandas as pd
from functools import reduce

#get all pred original logits from the group log file
def get_orig_logits(PATH):
    d = {}
    with open(PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)            
            d[record['filename']] = record
    return d

# class EvalLogits:
#     def __init__(self, full_video, model, groups, orig_stat, cls_idx, fill_type, show_mask=False):
#         self.full_video = full_video
#         self.model = model
#         self.groups = groups
#         self.orig_stat = orig_stat
#         self.cls_idx = cls_idx
#         self.show_mask = show_mask
#         self.fill_type = fill_type
#         self.N_CLS = 101

#         #prepare the all and none metrics
#         all_metrics, none_metrics = {},{}

#         # none_metrics['entropy'] = float(entropy(F.softmax(CONST.UCF_AVG_PRED,dim=0)))
#         # none_metrics['logit'] = float(CONST.UCF_AVG_PRED[cls_idx])
#         # for m in [1,3,5]:
#         #     none_metrics[f'margin_{m}'] = float(func.get_margin(CONST.UCF_AVG_PRED, cls_idx=cls_idx, k=m))
#         none_metrics['entropy'] = math.log(self.N_CLS)
#         none_metrics['logit'] = 1/self.N_CLS
#         for m in [1,3,5]:
#             none_metrics[f'margin_{m}'] = 0.0

#         logits = torch.tensor(orig_stat['logits'])
#         all_metrics['entropy'] = orig_stat['entropy']
#         all_metrics['logit'] = float(logits[cls_idx])
#         for m in [1,3,5]:
#             all_metrics[f'margin_{m}'] = float(func.get_margin(logits, cls_idx=cls_idx, k=m))

#         self.all_metrics, self.none_metrics = all_metrics, none_metrics

#     def eval_remove(self, idx_order):
#         metrics = {
#             'entropy': [0],
#             'margin1': [1],
#             'margin3': [1],
#             'margin5': [1],
#             'logit': [1]
#         }
#         mask = [1]*len(self.groups)
#         if self.show_mask:
#             print(mask)
#         for idx in idx_order:
#             mask[idx] = 0
#             if self.show_mask :
#                 print(mask)
#             if sum(mask) == 0:
#                 metrics['entropy'].append(1)
#                 metrics['margin1'].append(0)
#                 metrics['margin3'].append(0)
#                 metrics['margin5'].append(0)
#                 metrics['logit'].append(0)
#                 continue
#             if self.fill_type == 'past':
#                 g = [func.past_fill_all(mask, self.groups)]
#             elif self.fill_type == 'future':
#                 g = [func.future_fill_all(mask, self.groups)]
#             elif self.fill_type == 'middle':
#                 g = [func.hybrid_fill_all(mask, self.groups, 'middle')]
#             elif self.fill_type=='random':
#                 g = [func.hybrid_fill_all(mask, self.groups, 'random')]
#             elif self.fill_type=='late':
#                 g = [func.past_fill_all(mask, self.groups),
#                          func.future_fill_all(mask, self.groups)]

#             avg_ = {
#                 'entropy' : 0,
#                 'logit': 0,
#                 'margin1': 0,
#                 'margin3': 0,
#                 'margin5': 0,
#             }
#             n = 0
#             for g_ in g:
#                 vid_g = func.create_grouped_video(self.full_video, g_)
#                 stat = func.get_pred_stats(self.model, vid_g)
#                 # assert (stat['entropy'] - self.all_metrics['entropy'])/(self.none_metrics['entropy'] - self.all_metrics['entropy'])>0,'error'
#                 avg_['entropy'] += (stat['entropy'] - self.all_metrics['entropy'])/(self.none_metrics['entropy'] - self.all_metrics['entropy'])
#                 avg_['logit'] += (stat['logits'][self.cls_idx] - self.none_metrics['logit'])/(self.all_metrics['logit'] - self.none_metrics['logit'])
#                 for m in [1,3,5]:
#                     avg_[f'margin{m}'] += (stat[f'margin_{m}'] - self.none_metrics[f'margin_{m}'])/(self.all_metrics[f'margin_{m}'] - self.none_metrics[f'margin_{m}'])
#                 n += 1
#             for k in avg_:
#                 metrics[k].append(avg_[k]/n)

#         auc = {}
#         x = np.linspace(0, 1, len(metrics['entropy']))
#         # assert float(trapezoid(metrics['entropy'], x)) >=0 ,'entropy auc is negative'
#         auc['entropy'] = float(trapezoid(metrics['entropy'], x))
#         auc['logit'] = float(trapezoid(metrics['logit'], x))
#         for m in [1,3,5]:
#             # assert float(trapezoid(metrics[f'margin{m}'], x)) >=0 ,f'margin{m} auc is negative'
#             auc[f'margin{m}'] = float(trapezoid(metrics[f'margin{m}'], x))


#         # mask = [1,0,0]
#         # # mask[0]=0
#         # g_ = func.past_fill_all(mask, self.groups)
#         # vid_g = func.create_grouped_video(self.full_video, g_)
#         # stat = func.get_pred_stats(self.model, vid_g)
        
  
#         return {'list': metrics, 'auc': auc}
    
    
#     def eval_add(self, idx_order):
#         metrics = {
#             'entropy': [1],
#             'margin1': [0],
#             'margin3': [0],
#             'margin5': [0],
#             'logit': [0]
#         }
#         mask = [0]*len(self.groups)
#         if self.show_mask:
#             print(mask)
#         for idx in idx_order:
#             mask[idx] = 1
#             if self.show_mask :
#                 print(mask)
#             if sum(mask) == 0:
#                 metrics['entropy'].append(0)
#                 metrics['margin1'].append(1)
#                 metrics['margin3'].append(1)
#                 metrics['margin5'].append(1)
#                 metrics['logit'].append(1)
#                 continue
#             if self.fill_type == 'past':
#                 g = [func.past_fill_all(mask, self.groups)]
#             elif self.fill_type == 'future':
#                 g = [func.future_fill_all(mask, self.groups)]
#             elif self.fill_type == 'middle':
#                 g = [func.hybrid_fill_all(mask, self.groups, 'middle')]
#             elif self.fill_type=='random':
#                 g = [func.hybrid_fill_all(mask, self.groups, 'random')]
#             elif self.fill_type=='late':
#                 g = [func.past_fill_all(mask, self.groups),
#                          func.future_fill_all(mask, self.groups)]

#             avg_ = {
#                 'entropy' : 0,
#                 'logit': 0,
#                 'margin1': 0,
#                 'margin3': 0,
#                 'margin5': 0,
#             }
#             n = 0
#             for g_ in g:
#                 vid_g = func.create_grouped_video(self.full_video, g_)
#                 stat = func.get_pred_stats(self.model, vid_g)
#                 avg_['entropy'] += (stat['entropy'] - self.all_metrics['entropy'])/(self.none_metrics['entropy'] - self.all_metrics['entropy'])
#                 avg_['logit'] += (stat['logits'][self.cls_idx] - self.none_metrics['logit'])/(self.all_metrics['logit'] - self.none_metrics['logit'])
#                 for m in [1,3,5]:
#                     avg_[f'margin{m}'] += (stat[f'margin_{m}'] - self.none_metrics[f'margin_{m}'])/(self.all_metrics[f'margin_{m}'] - self.none_metrics[f'margin_{m}'])
#                 n += 1
#             for k in avg_:
#                 metrics[k].append(avg_[k]/n)

#         auc = {}
#         x = np.linspace(0, 1, len(metrics['entropy']))
#         auc['entropy'] = float(trapezoid(metrics['entropy'], x))
#         auc['logit'] = float(trapezoid(metrics['logit'], x))
#         for m in [1,3,5]:
#             auc[f'margin{m}'] = float(trapezoid(metrics[f'margin{m}'], x))

#         return {'list': metrics, 'auc': auc}

class EvalLogits:
    def __init__(self, full_video, model, groups, orig_stat, fill_type, show_mask=False):
        self.full_video = full_video
        self.model = model
        self.groups = groups
        self.orig_stat = orig_stat
        self.orig_logits = orig_stat['logits']
        self.orig_prob = F.softmax(torch.tensor(self.orig_logits),dim=0)
        self.pred_cls = np.argmax(np.array(self.orig_logits))
        self.show_mask = show_mask
        self.fill_type = fill_type
        self.N_CLS = 101

    def eval_remove(self, idx_order):
        metrics = {
            'prob': [self.orig_prob[self.pred_cls].item()],
            'logit': [self.orig_logits[self.pred_cls]]
        }

        mask = [1]*len(self.groups)
        if self.show_mask:
            print(mask)
        for idx in idx_order:
            mask[idx] = 0
            if self.show_mask :
                print(mask)

            if sum(mask) == 0:
                metrics['prob'].append(1/self.N_CLS)
                metrics['logit'].append(1/self.N_CLS)
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
                'prob' : 0,
                'logit': 0
            }
            n = 0
            for g_ in g:
                vid_g = func.create_grouped_video(self.full_video, g_)
                stat = func.get_pred_stats(self.model, vid_g)
                l = torch.tensor(stat['logits'])
                p = F.softmax(l,dim=0)
                avg_['prob'] += p[self.pred_cls].item()
                avg_['logit'] += l[self.pred_cls].item()
                n += 1
            for k in avg_:
                metrics[k].append(avg_[k]/n)

        auc = {}
        x = np.linspace(0, 1, len(metrics['prob']))
        auc['prob'] = float(trapezoid(metrics['prob'], x))
        auc['logit'] = float(trapezoid(metrics['logit'], x))
        
        return {'list': metrics, 'auc': auc}
    
    
    def eval_add(self, idx_order):
        metrics = {
            'prob': [1/self.N_CLS],
            'logit': [1/self.N_CLS]
        }

        mask = [0]*len(self.groups)
        if self.show_mask:
            print(mask)
        for idx in idx_order:
            mask[idx] = 1
            if self.show_mask :
                print(mask)

            if sum(mask) == len(self.groups):
                metrics['prob'].append(self.orig_prob[self.pred_cls].item())
                metrics['logit'].append(self.orig_logits[self.pred_cls])
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
                'prob' : 0,
                'logit': 0
            }
            n = 0
            for g_ in g:
                vid_g = func.create_grouped_video(self.full_video, g_)
                stat = func.get_pred_stats(self.model, vid_g)
                l = torch.tensor(stat['logits'])
                p = F.softmax(l,dim=0)
                avg_['prob'] += p[self.pred_cls].item()
                avg_['logit'] += l[self.pred_cls].item()
                n += 1
            for k in avg_:
                metrics[k].append(avg_[k]/n)

        auc = {}
        x = np.linspace(0, 1, len(metrics['prob']))
        auc['prob'] = float(trapezoid(metrics['prob'], x))
        auc['logit'] = float(trapezoid(metrics['logit'], x))
        
        return {'list': metrics, 'auc': auc}

def eval_UCF101(FILL_TYPE, IMP_PATH, GRP_PATH, OUT_PATH):
    # FILL_TYPE = 'past' # past, future, middle, random, late
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
            # if record['filename']!='v_YoYo_g07_c04': continue

            cls_idx = class_labels[record['filename'].split('_')[1].lower()]
            orig_stat = grp_stats[record['filename']]['original_stat']
            pred_cls = orig_stat['cls']

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

            sv_c = [s[pred_cls] for s in sv]
            asc_idx = [int(i) for i in np.argsort(sv_c)]

            # let groups have integer keys
            groups = {}
            for k in record['groups']:
                groups[int(k)] = record['groups'][k]

            el = EvalLogits(video, model, groups, orig_stat, FILL_TYPE)

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


def eval_UCF101_baseline(IMP_PATH, GRP_PATH, OUT_PATH):
    FILL_TYPE = 'late'

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
            imp = record['attribution']
            filename = record['filename']

            # let groups have integer keys
            groups = {}
            imp_values = []
            for k in grp_stats[filename]['groups']:
                if 'frames' in grp_stats[filename]['groups'][k]:
                    groups[int(k)] = grp_stats[filename]['groups'][k]['frames']
                else:
                    groups[int(k)] = grp_stats[filename]['groups'][k]
                    
                imp_values.append(imp[k])
            asc_idx = [int(i) for i in np.argsort(imp_values)]

            orig_stat = grp_stats[record['filename']]['original_stat']
            pred_cls = orig_stat['cls']

            p = ucf101dm.construct_vid_path_from_full(filename)
            video = ucf101dm.load_jpg_ucf101(p, n=0).permute(1,0,2,3)

            el = EvalLogits(video, model, groups, orig_stat, FILL_TYPE)

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
    EVAL_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\baselines\eval\IG_0.001.jsonl'
    # EVAL_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\baselines\eval\IG_0.001.jsonl'
    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    grp_stats = get_orig_logits(GRP_PATH)

    metrics = {
        'logit':0,
        'prob':0,
        'logit_norm':0,
        'prob_norm':0
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
    prob_auc = []
    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d_ = json.loads(line)
            
            # Dont consider cases where grouping changes the model prediction
            if grp_stats[d_['filename']]['original_stat']['cls'] != grp_stats[d_['filename']]['grp_pred_cls']:
                continue

            for k in set(d_.keys()) - {'filename'}:
                for m in d_[k]['auc']:
                    ar = np.array(d_[k]['list'][m])
                    ar_norm = (ar - ar.min())/(ar.max() - ar.min())
                    x = np.linspace(0, 1, len(ar))
                    auc_norm = float(trapezoid(ar_norm, x))
                    auc = float(trapezoid(ar, x))

                    d[k][m] += auc
                    d[k][f'{m}_norm'] += auc_norm
            n+=1

    for k in d:
        for m in d[k]:
            d[k][m]/=n
        
    metrics = ['prob', 'logit', 'prob_norm', 'logit_norm']

    for m in metrics:
        print(f'\n{m} AUC: ')
        for k in d:
            s = f'{k} : {d[k][m]}'
            print(s)
    print(f'n : {n}')

def normalize_list(l):
    l = np.array(l)
    l = (l-l.min())/(l.max() - l.min())
    return l

def plot_imp():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec

    ucf101dm = func.UCF101_data_model()

    EVAL_PATHS = {
        'best': r'C:\Users\lahir\Downloads\UCF101\analysis\shap\eval\exact_late_late_0.001.jsonl',
        'IG': r'C:\Users\lahir\Downloads\UCF101\analysis\baselines\eval\IG_0.001.jsonl',
        'gradcam': r'C:\Users\lahir\Downloads\UCF101\analysis\baselines\eval\gradcam_0.001.jsonl',
        'occlusion': r'C:\Users\lahir\Downloads\UCF101\analysis\baselines\eval\occlusion_0.001.jsonl'
    }
    IMP_PATHS = {
        'best': r'C:\Users\lahir\Downloads\UCF101\analysis\shap\exactSHAP_late_0.001.jsonl',
        'IG': r'C:\Users\lahir\Downloads\UCF101\analysis\baselines\0.001_IG.jsonl',
        'gradcam': r'C:\Users\lahir\Downloads\UCF101\analysis\baselines\0.001_gradcam.jsonl',
        'occlusion': r'C:\Users\lahir\Downloads\UCF101\analysis\baselines\0.001_occlusion.jsonl'
    }

    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    grp_stats = get_orig_logits(GRP_PATH)
    
    def get_metrics(EVAL_PATH):
        with open(EVAL_PATH, 'r', encoding='utf-8') as f:
            metrics = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d_ = json.loads(line)
                
                # Dont consider cases where grouping changes the model prediction
                if grp_stats[d_['filename']]['original_stat']['cls'] != grp_stats[d_['filename']]['grp_pred_cls']:
                    continue

                filename = d_['filename']
                # if filename != 'v_WalkingWithDog_g07_c03': continue

                def get_auc(d_, met):
                    ar = np.array(d_[met]['list']['prob'])
                    ar_norm = (ar - ar.min())/(ar.max() - ar.min())
                    x = np.linspace(0, 1, len(ar))
                    auc_norm = float(trapezoid(ar_norm, x))
                    return auc_norm
                metric = 0.5 * (get_auc(d_, 'rmv_asc')/get_auc(d_, 'rmv_dec') + get_auc(d_, 'add_dec')/get_auc(d_, 'add_asc'))
                metrics[filename] = metric
        return metrics
    
    IG_met = get_metrics(EVAL_PATHS['IG'])
    oc_met = get_metrics(EVAL_PATHS['occlusion'])
    gc_met = get_metrics(EVAL_PATHS['gradcam'])
    ex_met = get_metrics(EVAL_PATHS['best'])

    sort_idx = np.argsort(np.array([ex_met[k] for k in ex_met]))
    best_idx = sort_idx[-10:]
    worst_idx = sort_idx[:10]
    best_names = [list(ex_met.keys())[i] for i in best_idx]
    worst_names = [list(ex_met.keys())[i] for i in worst_idx]

    # get attributions
    def get_all_stats(names):
        imp_vals = {}
        for k in IMP_PATHS:
            imp_stats = get_orig_logits(IMP_PATHS[k])
            wanted_stats = {}
            for name in names:
                grp_cls = grp_stats[name]['grp_pred_cls']
                d_ = {}
                d_['grps'] = grp_stats[name]['groups']
                if k=='best':
                    d_['imp'] = [ar[grp_cls] for ar in imp_stats[name]['shapley_values'][0]]
                else:
                    d_['imp'] = [imp_stats[name]['attribution'][g] for g in d_['grps']]            
                wanted_stats[name] = d_
            imp_vals[k] = wanted_stats

        #make plots
        for name in names:
            
            groups = {}
            g=imp_vals['best'][name]['grps']
            for k in g:
                if 'frames' in g[k]:
                    f = g[k]['frames']
                else: 
                    f = []
                groups[int(k)] = f

            p = ucf101dm.construct_vid_path_from_full(name)
            video = ucf101dm.load_jpg_ucf101(p, n=0).permute(0,3,2,1)
            T,H,W,C = video.size()
            video = video.reshape(T*H,W,3).permute(1,0,2)
            video_norm = ((video - video.min()) / (video.max() - video.min())).numpy()
            
            fig = plt.figure(figsize=(20, 3))
            gs = GridSpec(2, 1, height_ratios=[1, 0.3], hspace=0.01)
            ax_frames = fig.add_subplot(gs[0])
            ax_frames.imshow(video_norm)
            ax_frames.axis('off')

            #plot rectangles around grouped frames
            x_vals = []
            for g in groups:
                frames = groups[g] + [g]
                # print(frames)
                frames.sort()
                x_vals.append(frames[0]*W + W*len(frames)*0.5)
                rect = Rectangle(
                    (frames[0]*W, 0),           # (x, y) - top-left corner
                    W*(len(frames)),              # width
                    H,             # height
                    linewidth=3,
                    edgecolor='red',
                    facecolor='none'
                )
                ax_frames.add_patch(rect)


            best_imp = normalize_list(imp_vals['best'][name]['imp'])
            best_ig = normalize_list(imp_vals['IG'][name]['imp'])
            best_gc = normalize_list(imp_vals['gradcam'][name]['imp'])
            best_oc = normalize_list(imp_vals['occlusion'][name]['imp'])

            ax_bars = fig.add_subplot(gs[1])
            total_width = video_norm.shape[1]
            ax_bars.set_xlim(0, total_width)
            ax_bars.axis('off')

            width = 10
            offset = 0.5
            x_vals = np.array(x_vals)
            bars = ax_bars.bar(x_vals, best_imp+offset, width, 
                       label='best', alpha=0.7, edgecolor='black')
            bars = ax_bars.bar(x_vals + width, best_ig+offset, width, 
                       label='ig', alpha=0.7, edgecolor='black')
            bars = ax_bars.bar(x_vals+2*width, best_gc+offset, width, 
                       label='gc', alpha=0.7, edgecolor='black')
            bars = ax_bars.bar(x_vals+3*width, best_oc+offset, width, 
                       label='oc', alpha=0.7, edgecolor='black')
            

            plt.show()

    get_all_stats(worst_names)

        



        





    pass




'''
correlations:

sv_middle_sv_past : 0.8921225773572666
sv_middle_sv_future : 0.8916806869270159
sv_middle_sv_late : 0.9249672775266151
sv_middle_sv_random : 0.9618467998371354
sv_past_sv_future : 0.8462951064300263
sv_past_sv_late : 0.956112907844077
sv_past_sv_random : 0.9125461436584569
sv_future_sv_late : 0.955814231520257
sv_future_sv_random : 0.9130876412734605
sv_late_sv_random : 0.9489035762684004

'''
def importance_correlation(GRP_PATH, IMP_PATH):
    grp_stats = get_orig_logits(GRP_PATH)
    imp_files = [file for file in os.listdir(IMP_PATH) if file.startswith('exact') and file.endswith('jsonl')]

    def get_sv(imp_file):
        file = os.path.join(IMP_PATH, imp_file)
        with open(file, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                line = line.strip()
                d_ = json.loads(line)
                filename = d_['filename']
                cls = grp_stats[filename]['stats']['cls']
                sv = [sv[cls] for sv in d_['shapley_values'][0]]
                if len(sv)==1: continue
                imp = imp_file.split('_')[1]
                data.append({'filename': filename, f'sv_{imp}': [sv[cls] for sv in d_['shapley_values'][0]]})
        df = pd.DataFrame(data)
        return df
    
    
    df_list = [get_sv(f) for f in imp_files]
    combined_df = reduce(lambda left, right: pd.merge(left, right, on='filename'), df_list)

    # calculate correlations
    list_columns = list(set(list(combined_df.columns))- {'filename'})
    summary_data = []
    for i in range(len(list_columns)):
        for j in range(i+1, len(list_columns)):
            col1 = list_columns[i]
            col2 = list_columns[j]
            
            correlations = []
            for idx in range(len(combined_df)):
                list1 = combined_df[col1].iloc[idx]
                list2 = combined_df[col2].iloc[idx]
                corr = np.corrcoef(list1, list2)[0, 1]
                correlations.append(corr)

            # np.where(np.isnan(correlations))[0]
            # combined_df.iloc[6]
            
            summary_data.append({
                'column_pair': f'{col1}_{col2}',
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'min_correlation': np.min(correlations),
                'max_correlation': np.max(correlations),
                'correlations_list': correlations
            })

    # print the mean correlations
    for sum in summary_data:
        name = sum['column_pair']
        corr = str(sum['mean_correlation'])
        print(f'{name} : {corr}')   
    

if __name__ == "__main__":
    # importance_correlation(r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl' ,r'C:\Users\lahir\Downloads\UCF101\analysis\shap')
    
    # IMP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\shap\exactSHAP_late_0.001.jsonl'
    # GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    # OUT_PATH = rf'C:\Users\lahir\Downloads\UCF101\analysis\shap\eval\exact_late_late_0.001.jsonl'
    # eval_UCF101(FILL_TYPE='late', IMP_PATH=IMP_PATH, GRP_PATH=GRP_PATH, OUT_PATH=OUT_PATH)
    plot_imp()
    # TYPE = 'IG'
    # IMP_PATH = rf'C:\Users\lahir\Downloads\UCF101\analysis\baselines\0.001_{TYPE}.jsonl'
    # GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    # OUT_PATH = rf'C:\Users\lahir\Downloads\UCF101\analysis\baselines\eval\{TYPE}_0.001.jsonl'
    # eval_UCF101_baseline(IMP_PATH, GRP_PATH, OUT_PATH)
