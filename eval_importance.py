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
from torch.utils.data import Subset

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


def eval_ssv2(FILL_TYPE, IMP_PATH, GRP_PATH, OUT_PATH):
    from dataloaders import ssv2
    from models.ssv2 import VJEPA2

    model = VJEPA2()
    model.eval()

    cls_list, path_list = ssv2.get_sampled_paths()
    nice_names = [Path(p).parent.name + '/' + Path(p).name for p in path_list]
    d = get_orig_logits(GRP_PATH)
    grp_stats = {}
    for k in d:
        grp_stats['/'.join(k.split('/')[-2:])] = d[k]


    #read existing data
    existing_names = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            n = 0
            for line in f:
                n+=1
                record = json.loads(line)
                existing_names.append(record['filename'])


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
            if record['filename'] in existing_names: continue

            orig_stat = grp_stats[record['filename']]['original_stat']
            pred_cls = orig_stat['cls']

            filename = record['filename']
            filename = filename.split('/')[-2] + '/' + filename.split('/')[-1]
            idx = nice_names.index(filename)
            p = path_list[idx]

            g = record['groups']
            groups = {}
            for k in g:
                if 'frames' in g[k]:
                    f = g[k]['frames']
                else: 
                    f = []
                groups[int(k)] = f

            video = model.video_from_path(p)['pixel_values_videos'][0,:]
            video = video.permute(1,0,2,3)

            sv = record['shapley_values'][0]

            sv_c = [s[pred_cls] for s in sv]
            asc_idx = [int(i) for i in np.argsort(sv_c)]

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


def eval_ssv2_baseline(FILL_TYPE, IMP_PATH, GRP_PATH, OUT_PATH):
    from dataloaders import ssv2
    from models.ssv2 import VJEPA2

    model = VJEPA2()
    model.eval()

    existing_names = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                existing_names.append(record['filename'])

    cls_list, path_list = ssv2.get_sampled_paths()
    nice_names = [Path(p).parent.name + '/' + Path(p).name for p in path_list]
    d = get_orig_logits(GRP_PATH)
    grp_stats = {}
    for k in d:
        grp_stats['/'.join(k.split('/')[-2:])] = d[k]

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
            if record['filename'] in existing_names: continue
            # if record['filename']!='v_YoYo_g07_c04': continue

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

            filename = filename.split('/')[-2] + '/' + filename.split('/')[-1]
            idx = nice_names.index(filename)
            p = path_list[idx]

            video = model.video_from_path(p)['pixel_values_videos'][0,:]
            video = video.permute(1,0,2,3)

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
    # EVAL_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\baselines\eval\partition_32_future_0.0001.jsonl'
    EVAL_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\baselines\eval\gradcam_0.001.jsonl'
    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    gs = get_orig_logits(GRP_PATH)

    # for ssv2
    grp_stats = {}
    for k in gs:
        grp_stats['/'.join(k.split('/')[-2:])] = gs[k]
    
    # unique filenames = 3783
    # files where n grps > 1 and grp pred aggrees with the all frames prediction = 3230
    # len(set(grp_stats.keys()))
    # c = 0
    # for k in grp_stats:
    #     if grp_stats[k]['original_stat']['cls'] != grp_stats[k]['grp_pred_cls']:
    #         continue
    #     if len(grp_stats[k]['groups']) <= 1:
    #         continue
    #     c+=1

    # part_data = get_orig_logits(EVAL_PATH)
    # n=0
    # for d in part_data:
    #     if grp_stats[d]['original_stat']['cls'] != grp_stats[d]['grp_pred_cls']:
    #         continue
    #     if len(grp_stats[d]['groups']) <= 1 :
    #         continue
    #     n+=1


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
    bad_grp = 0
    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d_ = json.loads(line)
            
            # Dont consider cases where grouping changes the model prediction
            if grp_stats[d_['filename']]['original_stat']['cls'] != grp_stats[d_['filename']]['grp_pred_cls']:
                bad_grp += 1
                # print(d_['filename'])
                continue
            if len(grp_stats[d_['filename']]['groups']) == 1:
                continue

            print(d_['filename'])

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
    print(f'bad grp: {bad_grp}')

def normalize_list(l):
    l = np.array(l)
    l = (l-l.min())/(l.max() - l.min())
    return l


def get_metrics(EVAL_PATH, grp_stats):
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
            
            rmv_asc = get_auc(d_, 'rmv_asc')
            rmv_dec = get_auc(d_, 'rmv_dec')
            rmv_rand = get_auc(d_, 'rmv_rand')
            add_dec = get_auc(d_, 'add_dec')
            add_asc = get_auc(d_, 'add_asc')
            add_rand = get_auc(d_, 'rmv_rand')
            
            
            metric = 0.5 * (rmv_asc/rmv_dec + add_dec/add_asc)
            met = {}
            met['metric'] = metric
            met['rmv_asc'] = rmv_asc
            met['rmv_dec'] = rmv_dec
            met['rmv_rand'] = rmv_rand
            met['add_asc'] = add_asc
            met['add_dec'] = add_dec
            met['add_rand'] = add_rand
            met['orig_pred'] = grp_stats[d_['filename']]['original_stat']['cls']
            met['grp_pred'] = grp_stats[d_['filename']]['grp_pred_cls']
            metrics[filename] = met
    return metrics
    
def plot_imp_ucf():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec

    ucf101dm = func.UCF101_data_model()
    inference_loader = ucf101dm.inference_loader
    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k
    
    cls_idx_path = r'C:\Users\lahir\Downloads\UCF101\analysis\class_idx.json'
    with open(cls_idx_path, 'r') as f:
        idx_data = json.load(f)

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
    
    IG_met = get_metrics(EVAL_PATHS['IG'], grp_stats)
    oc_met = get_metrics(EVAL_PATHS['occlusion'], grp_stats)
    gc_met = get_metrics(EVAL_PATHS['gradcam'], grp_stats)
    ex_met = get_metrics(EVAL_PATHS['best'], grp_stats)
    ex_met_ar = [ex_met[k]['metric'] for k in ex_met]


    sort_idx = np.argsort(np.array(ex_met_ar))
    best_idx = sort_idx[-10:]
    worst_idx = sort_idx[:10]
    best_names = [list(ex_met.keys())[i] for i in best_idx]
    worst_names = [list(ex_met.keys())[i] for i in worst_idx]

    # get attributions
    def get_all_stats(names, ex_met, class_names):
        out_path = r'C:\Users\lahir\Downloads\UCF101\analysis\plots\imp'

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

            idx = idx_data[name]
            inputs, targets = Subset(inference_loader.dataset, [idx])[0]
            video = inputs[0].permute(1,3,2,0)
            assert targets[0][0]==name , 'filename does not match!'

            T,H,W,C = video.size()
            video = video.reshape(T*H,W,3).permute(1,0,2)
            video_norm = ((video - video.min()) / (video.max() - video.min())).numpy()
            
            fig = plt.figure(figsize=(20, 2))
            gs = GridSpec(2, 2, 
              width_ratios=[1, 0.05],  # 85% for plot, 15% for text
              height_ratios=[1, 0.3], 
              hspace=0.00,
              wspace=0.00)
            
            ax_frames = fig.add_subplot(gs[0,0])
            ax_frames.imshow(video_norm)
            ax_frames.axis('off')

            ax_text = fig.add_subplot(gs[0, 1])  # gs[:, 1] spans both rows
            ax_text.axis('off')

            ra = ex_met[name]['rmv_asc']
            rd = ex_met[name]['rmv_dec']
            rr = ex_met[name]['rmv_rand']
            aa = ex_met[name]['add_asc']
            ad = ex_met[name]['add_dec']
            ar = ex_met[name]['add_rand'] 

            orig_pred = class_names[ex_met[name]['orig_pred']]
            grp_pred = class_names[ex_met[name]['grp_pred']]
            gt = name.split('_')[1]

            text = f'rmv_asc:{ra:.2f}\nrmv_dec:{rd:.2f} \nrmv_rand:{rr:.2f} \nGT:{gt} \npred:{orig_pred} \ngpred:{grp_pred}'

            ax_text.text(0.0, 0.5, 
                        text,
                        transform=ax_text.transAxes,
                        fontsize=7,
                        verticalalignment='center',
                        horizontalalignment='left',
                        fontfamily='monospace')

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

            ax_bars = fig.add_subplot(gs[1,0])
            total_width = video_norm.shape[1]
            ax_bars.set_xlim(0, total_width)
            ax_bars.set_ylim(0, 1.3)
            ax_bars.axis('off')


            width = 10
            offset = 0.1
            x_vals = np.array(x_vals)
            bars1 = ax_bars.bar(x_vals, best_imp+offset, width, 
                       label='Shap', alpha=0.7, edgecolor='black')
            bars2 = ax_bars.bar(x_vals + width, best_ig+offset, width, 
                       label='IG', alpha=0.7, edgecolor='black')
            bars3 = ax_bars.bar(x_vals+2*width, best_gc+offset, width, 
                       label='GC', alpha=0.7, edgecolor='black')
            bars4 = ax_bars.bar(x_vals+3*width, best_oc+offset, width, 
                       label='OC', alpha=0.7, edgecolor='black')
            

            # ax_bars.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
            ax_legend = fig.add_subplot(gs[1, 1])
            ax_legend.axis('off')

            # Create legend in bottom-right
            ax_legend.legend(handles=[bars1, bars2, bars3, bars4], 
                            labels=['Shap', 'IG', 'GC', 'OC'],
                            loc='center', 
                            fontsize=7,
                            frameon=False,
                            framealpha=0.9,
                            edgecolor='black',
                            ncol=1,
                            bbox_to_anchor=(0.3, 0.7))
            plt.tight_layout()
            plt.savefig(os.path.join(out_path,f'{name}.png'), bbox_inches='tight', pad_inches=0, dpi=300)

    get_all_stats(best_names, ex_met, class_names)

        



        





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
    
def imp_metric_vs_grouping(GRP_PATH, IMP_EVAL_PATH, PLT_PATH):
    import matplotlib.pyplot as plt

    grp_stats = get_orig_logits(GRP_PATH)
    imp_metrics = get_metrics(IMP_EVAL_PATH, grp_stats)

    # group length distribution
    # glens = [len(grp_stats[k]['groups']) for k in grp_stats.keys()]
    # plt.hist(glens)
    # plt.xticks(range(1, max(glens) + 1))
    # plt.xlabel('Num Groups')
    # plt.ylabel('Frequency')
    # plt.box(False) 
    # plt.savefig(os.path.join(PLT_PATH,'n_grp_dist.png'),bbox_inches='tight', pad_inches=0, dpi=300)

        
    imp = []
    logit_change, margin1_change, margin3_change, margin5_change = [],[],[],[]
    n_groups = []
    for filename in imp_metrics.keys():
        met = imp_metrics[filename]
        ng = len(grp_stats[filename]['groups'])
        if ng == 1: continue
        gstat = grp_stats[filename]['all_grp_stats']
        ostat = grp_stats[filename]['original_stat']
        change_stat = func.get_stat_change(ostat, gstat)

        if not grp_stats[filename]['correct']: continue

        imp.append(met)
        n_groups.append(ng)
        logit_change.append(change_stat['max_logit_change'])
        margin1_change.append(change_stat['margin_1_change'])
        margin3_change.append(change_stat['margin_3_change'])
        margin5_change.append(change_stat['margin_5_change'])

    # metric distribution
    # plt.hist(imp,bins=100)
    # plt.xlabel('Attribution Metric')
    # plt.ylabel('Frequency')
    # plt.box(False) 
    # plt.savefig(os.path.join(PLT_PATH,'imp_metric_dist.png'),bbox_inches='tight', pad_inches=0, dpi=300)
    
    # plt.scatter(n_groups, imp, s=5)
    # plt.xlabel('Num Groups')
    # plt.ylabel('Attribution Metric')
    # plt.box(False) 
    # plt.savefig(os.path.join(PLT_PATH,'ng_vs_metric.png'),bbox_inches='tight', pad_inches=0, dpi=300)

    # plt.scatter(logit_change, imp, s=5)
    # plt.xlabel('Logit Change')
    # plt.ylabel('Attribution Metric')
    # plt.box(False) 
    # plt.savefig(os.path.join(PLT_PATH,'lchange_vs_metric.png'),bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.xlim(-10,10)

    # plt.scatter(margin1_change, imp, s=5)
    # plt.xlabel('Margin 1 Change')
    # plt.ylabel('Attribution Metric')
    # plt.xlim(-10,3)
    # plt.box(False) 
    # plt.savefig(os.path.join(PLT_PATH,'M1change_vs_metric.png'),bbox_inches='tight', pad_inches=0, dpi=300)

    plt.scatter(margin5_change, imp, s=5)
    plt.xlabel('Margin 5 Change')
    plt.ylabel('Attribution Metric')
    plt.box(False) 
    plt.savefig(os.path.join(PLT_PATH,'M5change_vs_metric.png'),bbox_inches='tight', pad_inches=0, dpi=300)


def group_and_imp(model, video, gt_idx, GRP_THRESHOLD=1e-3):
    import group
    import importance
    from scipy.special import softmax

    FILL_METHOD = 'late'
    SHAP_METHOD = 'exact'
    N_SAMPLES = 0

    #********** grouping **************
    group_stats = group.group_frames(model, video, gt_idx, GRP_THRESHOLD)

    o_logits = np.array(group_stats['original_stat']['logits'])
    o_cls = np.argmax(o_logits)
    g_logits = group_stats['all_grp_stats']['logits']
    l_change = (o_logits[o_cls] - g_logits[o_cls])/(o_logits[o_cls])

    o_prob = softmax(group_stats['original_stat']['logits'])
    g_prob = softmax(group_stats['all_grp_stats']['logits'])
    p_change = (o_prob[o_cls] - g_prob[o_cls])/(o_prob[o_cls])

    m_changes = [] 
    orig_m = []
    grp_m = []
    for m in [1,3,5]:
        om = func.get_margin(torch.tensor(o_logits), o_cls, k=m)
        gm = func.get_margin(torch.tensor(g_logits), o_cls, k=m)
        m_change = (om-gm)/(om)
        orig_m.append(float(om))
        grp_m.append(float(gm))
        m_changes.append(m_change.item())

    o_entr = float(entropy(o_prob))
    g_entr = float(entropy(g_prob))


    #********** importance **************
    groups = {}
    for k in group_stats['groups']:
        if 'frames' in group_stats['groups'][k]:
            f = group_stats['groups'][k]['frames']
        else: 
            f = []
        groups[int(k)] = f

    # ex = importance.CalcSHAP(model, fill_method=FILL_METHOD, shap_method=SHAP_METHOD, N_SAMPLES=N_SAMPLES)

    # if SHAP_METHOD == 'exact':
    #     imp_values = ex.explain(video.permute(1,0,2,3), groups, check=True)
    #     imp_values = imp_values.values.tolist()[0]
    #     imp_values = [val[o_cls] for val in imp_values]
    # if SHAP_METHOD == 'kernel':
    #     imp_values = ex.explain_kernel(video, groups, check=True)
    # if SHAP_METHOD == 'partition':
    #     imp_values = ex.explain_partition(video, groups, check=True)

    # #calc imp metrics
    # el = EvalLogits(video, model, groups, group_stats['original_stat'], FILL_METHOD)

    # asc_idx = [int(i) for i in np.argsort(imp_values)]

    # results = {
    #     'rmv_asc': el.eval_remove(asc_idx),
    #     'rmv_dec': el.eval_remove(asc_idx[::-1]),
    #     'rmv_rand': el.eval_remove(random.sample(asc_idx, len(asc_idx))),
    #     'rmv_lr': el.eval_remove(sorted(asc_idx)),
    #     'rmv_rl': el.eval_remove(sorted(asc_idx)[::-1]),
    #     'add_asc': el.eval_add(asc_idx),
    #     'add_dec': el.eval_add(asc_idx[::-1]),
    #     'add_rand': el.eval_add(random.sample(asc_idx, len(asc_idx))),
    #     'add_lr': el.eval_add(sorted(asc_idx)),
    #     'add_rl': el.eval_add(sorted(asc_idx)[::-1])
    # }

    output = {}
    # output['auc'] = results
    output['grp_metrics'] = {
        'groups': groups,
        'orig_logit': o_logits[o_cls],
        'orig_prob': o_prob[o_cls],
        'grp_logit': g_logits[o_cls],
        'grp_prob': g_prob[o_cls],
        'orig_m': orig_m,
        'grp_m': grp_m,
        'orig_entr': o_entr,
        'grp_entr': g_entr,
        'logit_change': float(l_change),
        'prob_change': float(p_change),
        'm_changes' : m_changes,
        'gt_cls': gt_idx,
        'pred_cls': group_stats['original_stat']['cls'],
        'grp_cls': group_stats['grp_pred_cls']
    }


    return output

def plot_iterative_grouping(model, video, out_path, gt_cls, class_names, THR=1e-3):

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    dpi = 100
    
    full_vid_path = os.path.join(out_path,'full.png')

    C,T,H,W = video.size()
    img_width_px = W * T        # total width of image strip
    img_height_px = H           # image height
    text_width_px = 200         # width reserved for text
    
    fig_width = (img_width_px + text_width_px) / dpi
    fig_height = img_height_px / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = GridSpec(
        1, 2,
        width_ratios=[img_width_px, text_width_px],
        wspace=0.005
    )
    ax_img = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    ax_img.set_xlim(0, W*T)  # Show from before first to after last
    ax_img.set_ylim(H, 0)  # Show full height with some padding
    ax_img.axis('off') 
    ax_text.axis('off') 

    out = group_and_imp(model, video, gt_cls, GRP_THRESHOLD=THR)
    pred_cls = out['grp_metrics']['pred_cls']
    pred_cls_str = class_names[pred_cls]
    gt_cls_str = class_names[gt_cls]

    if not os.path.exists(full_vid_path):
        # show the video
        videop = video.permute(2,1,3,0)
        W,T,H,C = videop.size()
        videop = videop.reshape(W,T*H,3)
        videop_norm = ((videop - videop.min()) / (videop.max() - videop.min())).numpy()

        ax_img.imshow(videop_norm)
        plt.axis('off')

        L = float(out['grp_metrics']['orig_logit'])
        p = out['grp_metrics']['orig_prob']
        m1 = float(out['grp_metrics']['orig_m'][0])
        m3 = float(out['grp_metrics']['orig_m'][1])
        m5 = float(out['grp_metrics']['orig_m'][2])
        entr = float(out['grp_metrics']['orig_entr'])

        text = f'GT={gt_cls_str} \npred={pred_cls_str}\nL={L:.2f} P={p:.2f} \nm1={m1:.2f} m3={m3:.2f} \nm5={m5:.2f} e={entr:.2f}'

        ax_text.text(
            0.0, 0.5,
            text,
            transform=ax_text.transAxes,
            fontsize=9,
            verticalalignment='center',
            horizontalalignment='left'
        )
        plt.savefig(full_vid_path, bbox_inches='tight', pad_inches=0, dpi=dpi)


    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = GridSpec(
        1, 2,
        width_ratios=[img_width_px, text_width_px],
        wspace=0.005
    )
    ax_img = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    ax_img.set_xlim(0, W*T)  # Show from before first to after last
    ax_img.set_ylim(H, 0)  # Show full height with some padding
    ax_img.axis('off') 
    ax_text.axis('off') 

    groups = out['grp_metrics']['groups']
    kframes = list(groups.keys())

    C,T,H,W = video.size()
    #calculate x pos of key frames
    xpos = [k*W for k in kframes]

    for i in range(len(kframes)):
        k = kframes[i]
        frame = video[:,k,:].permute(1,2,0)
        frame = ((frame - frame.min()) / (frame.max() - frame.min())).numpy()
        ax_img.imshow(frame, extent=[xpos[i], xpos[i]+W, 
                                       0 + H, 0])

    #display metrics
    l = out['grp_metrics']['grp_logit']
    p = out['grp_metrics']['grp_prob']
    m1 = out['grp_metrics']['grp_m'][0]
    m3 = out['grp_metrics']['grp_m'][1]
    m5 = out['grp_metrics']['grp_m'][2]
    e = out['grp_metrics']['grp_entr']
    grp_cls = out['grp_metrics']['grp_cls']
    grp_cls_str = class_names[grp_cls]

    text = f'Thr={THR} \nL={l:.2f} P={p:.2f} \nm1={m1:.2f} m3={m3:.2f} \nm5={m5:.2f} e={e:.2f} \ngrp={grp_cls_str}'

    ax_text.text(
        0.0, 0.5,
        text,
        transform=ax_text.transAxes,
        fontsize=9,
        verticalalignment='center',
        horizontalalignment='left'
    )
    ax_text.axis('off')

    fig.savefig(os.path.join(out_path,f'{THR}.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close('all')
    print(f'orig_pred : {pred_cls_str}  grp_pred: {grp_cls_str}')

'''
iterate different thresholds for grouping and plot the logits and other metrics as we do
'''
def iterative_grouping_ucf(filename):
    from torch.utils.data import Subset

    cls_idx_path = r'C:\Users\lahir\Downloads\UCF101\analysis\class_idx.json'
    with open(cls_idx_path, 'r') as f:
        idx_data = json.load(f)
    idx = idx_data[filename]

    out_path = r'C:\Users\lahir\Downloads\UCF101\analysis\plots\grouping\iterative_grouping'
    out_path = os.path.join(out_path,filename)
    os.makedirs(out_path,exist_ok=True)

    ucf101dm = func.UCF101_data_model()
    # p = ucf101dm.construct_vid_path_from_full(filename)
    # video = ucf101dm.load_jpg_ucf101(p, n=0).permute(0,3,2,1)
    # video = video.permute(3,0,1,2)

    model = ucf101dm.model
    inference_loader = ucf101dm.inference_loader
    inference_class_names = ucf101dm.inference_class_names
    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k
    #****************************************************************************

    inputs, targets = Subset(inference_loader.dataset, [idx])[0]
    video = inputs[0]
    assert targets[0][0]==filename , 'filename does not match!'

    # for idx, batch in enumerate(inference_loader):
    #     print(f'{idx/len(inference_loader)*100:.2f} % is done.', end='\r')
    #     # if idx==40: break
    #     inputs, targets = batch
    #     if targets[0][0] != filename: continue
    #     video = inputs[0,:]
    #     break
    
    gt_cls = class_labels[filename.split('_')[1].lower()]

    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    gs = get_orig_logits(GRP_PATH)

    gs[filename]['grp_pred_cls']
    gs[filename]['original_stat']['cls']
    gs[filename]['groups'].keys()

    plot_iterative_grouping(model, video, out_path, gt_cls, class_names, THR=-1)

def ucf_dataset_explore():

    OUT_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\class_idx.json'
    
    ucf101dm = func.UCF101_data_model()
    inference_loader = ucf101dm.inference_loader

    d = {}
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.2f} % is done.', end='\r')
        inputs, targets = batch
        d[targets[0][0]] = idx

    with open(OUT_PATH, 'w') as f:
        json.dump(d, f)


if __name__ == "__main__":
    # importance_correlation(r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl' ,r'C:\Users\lahir\Downloads\UCF101\analysis\shap')
    
    IMP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\shap\framewise\frame_partition_32_late_0.001.jsonl'
    GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    OUT_PATH = rf'C:\Users\lahir\Downloads\UCF101\analysis\shap\framewise\eval\frame_partition_32_late_late_0.001.jsonl'
    eval_UCF101(FILL_TYPE='late', IMP_PATH=IMP_PATH, GRP_PATH=GRP_PATH, OUT_PATH=OUT_PATH)

    # plot_imp()

    # TYPE = 'IG'
    # IMP_PATH = rf'C:\Users\lahir\Downloads\UCF101\analysis\baselines\0.001_{TYPE}.jsonl'
    # GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    # OUT_PATH = rf'C:\Users\lahir\Downloads\UCF101\analysis\baselines\eval\{TYPE}_0.001.jsonl'
    # eval_UCF101_baseline(IMP_PATH, GRP_PATH, OUT_PATH)

    
    # IMP_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\baselines\0.0001_gradcam.jsonl'
    # GRP_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\groups\groups_0.0001.jsonl'
    # OUT_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\baselines\eval\occlusion_future_0.001.jsonl'
    # eval_ssv2_baseline(FILL_TYPE='future', IMP_PATH=IMP_PATH, GRP_PATH=GRP_PATH, OUT_PATH=OUT_PATH)

    # IMP_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\shap\partition_32_future_0.0001.jsonl'
    # GRP_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\groups\groups_0.0001.jsonl'
    # OUT_PATH = r'C:\Users\lahir\Downloads\partition_32_future_0.0001.jsonl'
    # eval_ssv2(FILL_TYPE='future', IMP_PATH=IMP_PATH, GRP_PATH=GRP_PATH, OUT_PATH=OUT_PATH)

    # GRP_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl'
    # IMP_EVAL_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\shap\eval\exact_late_late_0.001.jsonl'
    # PLT_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\shap\eval\plots'
    # imp_metric_vs_grouping(GRP_PATH, IMP_EVAL_PATH, PLT_PATH)

    # ucf_dataset_explore()

    # iterative_grouping_ucf('v_YoYo_g04_c03')


    # avg_stat_ucf()

    # plot_imp_ucf()
