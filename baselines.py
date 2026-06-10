import numpy as np
import torch
import torch.nn as nn
from captum.attr import GuidedGradCam, IntegratedGradients, FeatureAblation, Occlusion
from CONST import UCF_INP_SHAPE
import func
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

torch.manual_seed(123)
np.random.seed(123)


def show_overlay(video, attribution, alpha=0.5):
    """
    video: tensor of shape (3, 16, 112, 112) - C, T, H, W
    attribution: tensor of shape (3, 16, 112, 112) - C, T, H, W
    """
    video_np = video.cpu().numpy()
    attr_np = attribution.cpu().numpy()
    
    # Normalize video to [0, 1]
    video_min = video_np.min(axis=(1,2,3), keepdims=True)
    video_max = video_np.max(axis=(1,2,3), keepdims=True)
    video_norm = (video_np - video_min) / (video_max - video_min + 1e-8)
    
    # Normalize attribution to [-1, 1]
    attr_abs_max = np.abs(attr_np).max()
    attr_norm = attr_np / (attr_abs_max + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    def update(frame):
        for ax in axes:
            ax.clear()
            ax.axis('off')
        
        # Original
        frame_orig = np.transpose(video_norm[:, frame], (1, 2, 0))
        axes[0].imshow(frame_orig)
        axes[0].set_title(f'Original - Frame {frame}')
        
        # Overlay
        attr_overlay = np.transpose(attr_norm[:, frame], (1, 2, 0))
        pos_attr = np.maximum(attr_overlay, 0).mean(axis=-1)  # Positive in red
        neg_attr = np.abs(np.minimum(attr_overlay, 0)).mean(axis=-1)  # Negative in blue
        
        overlay_color = np.zeros_like(frame_orig)
        overlay_color[..., 0] = pos_attr  # Red
        overlay_color[..., 2] = neg_attr  # Blue
        
        blended = (1 - alpha) * frame_orig + alpha * overlay_color
        blended = np.clip(blended, 0, 1)
        
        axes[1].imshow(blended)
        axes[1].set_title('Overlay (Red: +, Blue: -)')
    
    ani = animation.FuncAnimation(fig, update, frames=video.shape[1], 
                                  interval=200, blit=False)
    plt.tight_layout()
    plt.show()
    return ani
    
class Baseline():
    def __init__(self, model, method):
        self.model = model.to('cuda')
        if method == 'IG':
            self.interpr = IntegratedGradients(self.model)
            self.baseline = torch.zeros(*UCF_INP_SHAPE).to('cuda')
        elif method == 'gradcam':
            self.interpr = GuidedGradCam(self.model, self.model.layer4)
        elif method == 'feat_abl':
            self.interpr = FeatureAblation(self.model)
        elif method == 'occlusion':
            self.interpr = Occlusion(self.model)
        self.method = method

    
    def frame_attribute(self, input, target):
        if self.method == 'IG':
            attributions = self.interpr.attribute(input.to('cuda')[None,:], self.baseline.to('cuda')[None,:], target=target)
        if self.method == 'gradcam':
            attributions = self.interpr.attribute(input.to('cuda')[None,:], target=target)
        elif self.method == 'feat_abl': # takes too much time
            attributions = self.interpr.attribute(input.to('cuda')[None,:], target=target)
        elif self.method == 'occlusion':
            h,w = input.size(2),input.size(3)
            attributions = self.interpr.attribute(input[None,:].to('cuda'), target=target, sliding_window_shapes=(1,1,h,w))
        # show_overlay(input, attributions[0,:].detach().cpu())
        # get frame wise attributions
        attributions = torch.mean(attributions, dim=(1,3,4))[0,:]
        #normalize attributions
        # attributions = (attributions - torch.min(attributions)) / (torch.max(attributions)-torch.min(attributions) + 1e-8)

        return attributions.detach().cpu()

import os

def calc_imp_UCF(GRP_PATH, OUT_PATH, INTERPR_METHOD='IG'):
    #create outout file
    thr = Path(GRP_PATH).stem.split('_')[-1]
    filename = f'{thr}_{INTERPR_METHOD}.jsonl'
    OUT_PATH = os.path.join(OUT_PATH, filename)

    #****************************************************************************
    # the model and the data loader
    #****************************************************************************
    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    baseline = Baseline(model, method=INTERPR_METHOD)

    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k

    n=0
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.1f}% is done.', end='\r')
            n+=1
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            filename = record['filename']
            p = ucf101dm.construct_vid_path_from_full(filename)
            g = record['groups']
            groups = {}
            for k in g:
                if 'frames' in g[k]:
                    f = g[k]['frames']
                else: 
                    f = []
                groups[int(k)] = f
            
            video = ucf101dm.load_jpg_ucf101(p, n=0)
            video_g = func.create_grouped_video(video.permute(1,0,2,3), groups) # create grouped video
            pred_cls = record['grp_pred_cls']
            attr = baseline.frame_attribute(video_g, target=pred_cls)

            # per-group attributions
            grp_attr = {}
            s = 0
            for k in groups:
                f_idx = [k] + groups[k]
                m = float(attr[f_idx].mean())
                grp_attr[k] = m
                s += m
            for k in groups:
                grp_attr[k]/=s

            d={}
            d['filename'] = filename
            d['attribution'] = grp_attr
            d['correct'] = record['correct']
            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(d) + '\n')

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

def calc_imp_ssv2(GRP_PATH, OUT_PATH, INTERPR_METHOD='IG'):
    from dataloaders import ssv2
    from models.ssv2 import VJEPA2
    import time

    #create outout file
    thr = Path(GRP_PATH).stem.split('_')[-1]
    filename = f'{thr}_{INTERPR_METHOD}.jsonl'
    OUT_PATH = os.path.join(OUT_PATH, filename)

    exist_files = []
    if os.path.exists(OUT_PATH):
        data = get_orig_logits(OUT_PATH)
        exist_files = list(data.keys())

    # print('in ssv2 shape calc')
    model = VJEPA2()
    model.eval()
    class_names = list(model.label2id.keys())
    baseline = Baseline(model, method=INTERPR_METHOD)

    cls_list, path_list = ssv2.get_sampled_paths()
    n_files = len(path_list)
    nice_names = [Path(p).parent.name + '/' + Path(p).name for p in path_list]

    n=0
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        start_time = time.time()
        for line in f:
            end_time = time.time()
            elapsed = end_time - start_time

            print(f'{n/line_count*100:.1f}% is done. Time passed: {elapsed:.2f}s', end='\r')
            n+=1

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            
            # ignore if the grouping changed the prediction
            if record['grp_pred_cls'] != record['original_stat']['cls']: continue

            filename = record['filename']
            filename = filename.split('/')[-2] + '/' + filename.split('/')[-1]
            if filename in exist_files: continue

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
            if len(groups)==1: continue

            video = model.video_from_path(p)['pixel_values_videos'][0,:]
            video_g = func.create_grouped_video(video.permute(1,0,2,3), groups)
            pred_cls = record['grp_pred_cls']
            attr = baseline.frame_attribute(video_g, target=pred_cls)
            # per-group attributions
            grp_attr = {}
            s = 0
            for k in groups:
                f_idx = [k] + groups[k]
                m = float(attr[f_idx].mean())
                grp_attr[k] = m
                s += m
            for k in groups:
                grp_attr[k]/=s

            d={}
            d['filename'] = filename
            d['attribution'] = grp_attr
            d['correct'] = record['correct']
            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(d) + '\n')


if __name__ == "__main__":
    GRP_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\groups\groups_0.0001.jsonl'
    OUT_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\baselines'
    calc_imp_ssv2(GRP_PATH, OUT_PATH, INTERPR_METHOD='occlusion')