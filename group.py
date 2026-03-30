import torch
import os
from glob import glob
from PIL import Image
import func

ucf101dm = func.UCF101_data_model()
model = ucf101dm.model
inference_loader = ucf101dm.inference_loader
inference_class_names = ucf101dm.inference_class_names
class_names = ucf101dm.inference_class_names
class_labels = {}
for k in class_names.keys():
    cls_name = class_names[k]
    class_labels[cls_name.lower()] = k


def group_frames(model, video, gt_idx):
    #get original pred
    with torch.inference_mode():
        pred = model(video[None,:])
        pred_idx = torch.argmax(pred,dim=1).item()
        pred_logit = pred[0][pred_idx].item()

    func.get_pred_stats(model, video, gt_idx, pred_logit)
    model(video).shape
    
    pass


def group_frames_loader():
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        cls = [class_labels[t[0].split('_')[1].lower()] for t in targets]
        video = inputs[0,:]
        gt_idx = class_labels[targets[0][0].split('_')[1].lower()]
        group_frames(model, video, gt_idx)
        pass

if __name__ == '__main__':
    group_frames_loader()