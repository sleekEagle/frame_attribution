import numpy as np
import torch
import torch.nn as nn
from captum.attr import GuidedGradCam, IntegratedGradients, GuidedGradCam, FeatureAblation
from CONST import UCF_INP_SHAPE
import func
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

torch.manual_seed(123)
np.random.seed(123)

UCF_PATH = r'C:\Users\lahir\Downloads\UCF101\analysis\groups_0.001.jsonl'
OUT_DICT = r'C:\Users\lahir\Downloads\UCF101\analysis\baselines'


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
        self.method = method

    
    def frame_attribute(self, input, target):
        if self.method == 'IG':
            attributions = self.interpr.attribute(input.to('cuda')[None,:], self.baseline.to('cuda')[None,:], target=target)
        if self.method == 'gradcam':
            attributions = self.interpr.attribute(input.to('cuda')[None,:], target=target)
        elif self.method == 'feat_abl': # takes too much time
            attributions = self.interpr.attribute(input.to('cuda')[None,:], target=target)
        # show_overlay(input, attributions[0,:].detach().cpu())
        attributions = torch.mean(attributions, dim=(1,3,4))[0,:]
        #normalize attributions
        attributions = (attributions - torch.min(attributions)) / (torch.max(attributions)-torch.min(attributions) + 1e-8)

        return attributions.detach().cpu()

from pathlib import Path

def calc_imp():
    #create outout file
    INTERPR_METHOD = 'IG'
    thr = Path(UCF_PATH).stem.split('_')[-1]
    OUT_PATH = rf'{OUT_DICT}\{thr}_{INTERPR_METHOD}.jsonl'

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
    with open(UCF_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))
    with open(UCF_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.1f}% is done.', end='\r')
            n+=1
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            filename = record['filename']
            p = ucf101dm.construct_vid_path_from_full(filename)
            video = ucf101dm.load_jpg_ucf101(p, n=0)
            attr = baseline.frame_attribute(video.permute(1,0,2,3), target=class_labels[filename.split('_')[1].lower()])

            d={}
            d['filename'] = filename
            d['attribution'] = attr.tolist()
            with open(OUT_PATH, 'a') as f:
                f.write(json.dumps(d) + '\n')

if __name__ == "__main__":
    calc_imp()