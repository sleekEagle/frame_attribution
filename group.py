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

def group_frames():
    pass

if __name__ == '__main__':
    group_frames()